import io
import os
import threading
import time
import wave
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from snac import SNAC

load_dotenv()

# Try to enable torch.compile if PyTorch 2.0+ is available
TORCH_COMPILE_AVAILABLE = False
try:
    if hasattr(torch, "compile"):
        TORCH_COMPILE_AVAILABLE = True
        print("PyTorch 2.0+ detected, torch.compile is available")
except Exception as e:
    print(f"Error checking torch.compile: {e}")
    pass

# Try to enable CUDA graphs if available
CUDA_GRAPHS_AVAILABLE = False
try:
    if torch.cuda.is_available() and hasattr(torch.cuda, "make_graphed_callables"):
        CUDA_GRAPHS_AVAILABLE = True
        print("CUDA graphs support is available")
except Exception as e:
    print(f"Error checking CUDA graphs: {e}")
    pass

# Check if CUDA is available and set device accordingly
snac_device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {snac_device}")


# Create a pool of models for concurrent processing
class ModelPool:
    def __init__(self, size=10):
        self.size = size
        self.models: List[SNAC] = []
        self.locks: List[threading.Lock] = []
        self.cuda_streams: List[Optional[torch.cuda.Stream]] = []
        self.in_use = [False] * size
        self.pool_lock = threading.Lock()

        print(f"Initializing model pool with {size} models...")
        for i in range(size):
            model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            model = model.to(snac_device)
            self.models.append(model)
            self.locks.append(threading.Lock())

            # Create a CUDA stream for this model if using CUDA
            if snac_device == "cuda":
                self.cuda_streams.append(torch.cuda.Stream())
            else:
                self.cuda_streams.append(None)

        print(f"Model pool initialized with {size} models")

    def get_model(self):
        """Get an available model with its lock and stream"""
        with self.pool_lock:
            # First try to find an unused model
            for i in range(self.size):
                if not self.in_use[i]:
                    self.in_use[i] = True
                    return i, self.models[i], self.locks[i], self.cuda_streams[i]

            # If all models are in use, wait for one to become available
            print("All models are in use, waiting for one to become available...")
            while True:
                # Check if any model has become available
                for i in range(self.size):
                    if not self.in_use[i]:
                        self.in_use[i] = True
                        return i, self.models[i], self.locks[i], self.cuda_streams[i]
                # Wait a bit before checking again
                time.sleep(0.1)

    def release_model(self, index):
        """Release a model back to the pool"""
        with self.pool_lock:
            self.in_use[index] = False


# Initialize the model pool with 10 models (configurable)
MODEL_POOL_SIZE = int(os.environ.get("MODEL_POOL_SIZE", "10"))
model_pool = ModelPool(size=MODEL_POOL_SIZE)


def create_wav_header(sample_rate: int, channels: int, sample_width: int) -> bytes:
    """Creates a WAV header with placeholder sizes."""
    # Using BytesIO to build the header in memory
    f = io.BytesIO()
    wf = wave.open(f, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(sample_rate)
    # Write 0 frames - this forces wave module to write the header structure
    # The chunk sizes (RIFF overall size, data chunk size) will be placeholders
    # like 0 or a small default, which is often sufficient for streaming players.
    wf.writeframes(b"")
    wf.close()  # Close the wave writer *before* getting the value
    return f.getvalue()


def convert_to_audio(multiframe, count, model_index):
    """
    Optimized version of convert_to_audio that eliminates inefficient tensor operations
    and reduces CPU-GPU transfers for much faster inference on high-end GPUs.
    """

    model = model_pool.models[model_index]
    cuda_stream = model_pool.cuda_streams[model_index]

    if len(multiframe) < 7:
        return None

    num_frames = len(multiframe) // 7
    frame = multiframe[: num_frames * 7]

    # Pre-allocate tensors instead of incrementally building them
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)

    # Use vectorized operations where possible
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)

    # Direct indexing is much faster than concatenation in a loop
    for j in range(num_frames):
        idx = j * 7

        # Code 0 - single value per frame
        codes_0[j] = frame_tensor[idx]

        # Code 1 - two values per frame
        codes_1[j * 2] = frame_tensor[idx + 1]
        codes_1[j * 2 + 1] = frame_tensor[idx + 4]

        # Code 2 - four values per frame
        codes_2[j * 4] = frame_tensor[idx + 2]
        codes_2[j * 4 + 1] = frame_tensor[idx + 3]
        codes_2[j * 4 + 2] = frame_tensor[idx + 5]
        codes_2[j * 4 + 3] = frame_tensor[idx + 6]

    # Reshape codes into expected format
    codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

    # Check tokens are in valid range
    if (
        torch.any(codes[0] < 0)
        or torch.any(codes[0] > 4096)
        or torch.any(codes[1] < 0)
        or torch.any(codes[1] > 4096)
        or torch.any(codes[2] < 0)
        or torch.any(codes[2] > 4096)
    ):
        return None

    # Use CUDA stream for parallel processing if available
    stream_ctx = (
        torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()
    )

    with stream_ctx, torch.inference_mode():
        # Decode the audio
        audio_hat = model.decode(codes)

        # Extract the relevant slice and efficiently convert to bytes
        # Keep data on GPU as long as possible
        audio_slice = audio_hat[:, :, 2048:4096]

        # Process on GPU if possible, with minimal data transfer
        if snac_device == "cuda":
            # Scale directly on GPU
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            # Only transfer the final result to CPU
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            # For non-CUDA devices, fall back to the original approach
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

    return audio_bytes


# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Use a single global cache for token processing
token_id_cache = {}
MAX_CACHE_SIZE = 10000  # Increased cache size for better performance


def turn_token_into_id(token_string, index):
    """
    Optimized token-to-ID conversion with caching.
    This is the definitive implementation used by both inference.py and speechpipe.py.

    Args:
        token_string: The token string to convert
        index: Position index used for token offset calculation

    Returns:
        int: Token ID if valid, None otherwise
    """
    # Check cache first (significant speedup for repeated tokens)
    cache_key = (token_string, index % 7)
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]

    # Early rejection for obvious non-matches
    if CUSTOM_TOKEN_PREFIX not in token_string:
        return None

    # Process token
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)

    if last_token_start == -1:
        return None

    last_token = token_string[last_token_start:]

    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
        return None

    try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)

        # Cache the result if it's valid
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = token_id

        return token_id
    except (ValueError, IndexError):
        return None
