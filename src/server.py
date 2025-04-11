import asyncio
import io
import os
import time
import wave

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference import DEFAULT_VOICE, SAMPLE_RATE, generate_tokens_from_api_async
from speechpipe import tokens_decoder


# Function to ensure .env file exists
def ensure_env_file_exists():
    """Create a default .env file if one doesn't exist"""
    if not os.path.exists(".env") and os.path.exists(".env.example"):
        try:
            # Copy .env.example to .env
            with open(".env.example", "r") as example_file:
                with open(".env", "w") as env_file:
                    env_file.write(example_file.read())
            print("✅ Created default configuration file at .env")
        except Exception as e:
            print(f"⚠️ Error creating default .env file: {e}")


# Ensure .env file exists before loading environment variables
ensure_env_file_exists()

# Load environment variables from .env file
load_dotenv(override=True)

# Create FastAPI app
app = FastAPI(
    title="Orpheus-FASTAPI",
    description="High-performance Text-to-Speech server using Orpheus-FASTAPI",
    version="1.0.0",
)


# API models
class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: str = "wav"
    speed: float = 1.0


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


# OpenAI-compatible API endpoint
@app.post("/v1/audio/speech")
async def create_speech_api(request: SpeechRequest):
    """
    Generate speech from text using the Orpheus TTS model.
    Compatible with OpenAI's /v1/audio/speech endpoint.

    For longer texts (>1000 characters), batched generation is used
    to improve reliability and avoid truncation issues.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")

    # Create a queue to buffer audio chunks between generation and streaming
    audio_queue = asyncio.Queue()  # Default infinite size
    all_audio_segments = []
    start_time = time.time()

    # Define the audio generation task (producer)
    async def audio_producer():
        try:
            print("Starting audio generation...")
            token_gen = generate_tokens_from_api_async(
                prompt=request.input,
                voice=request.voice,
            )
            samples_gen = tokens_decoder(token_gen)

            async for audio_chunk in samples_gen:
                if audio_chunk:
                    await audio_queue.put(audio_chunk)
                    # Optional: Add slight delay if producer is too fast
                    # await asyncio.sleep(0.01)

            print("Audio generation complete. Signalling end.")
            await audio_queue.put(None)  # Signal end of stream

        except Exception as e:
            print(f"⚠️ Error during audio generation: {e}")
            # Put the exception or a special error marker if consumer needs to know
            # For simplicity, we just signal end here, but error handling could be more robust.
            await audio_queue.put(None)  # Ensure consumer doesn't hang
            # Optionally, re-raise or log more details
            # raise # This would terminate the task

    # Start the audio generation task in the background
    producer_task = asyncio.create_task(audio_producer())

    # Define the streaming generator (consumer)
    async def stream_generator():
        # 1. Yield the WAV header first
        channels = 1
        sample_width = 2  # 16-bit PCM
        header = create_wav_header(SAMPLE_RATE, channels, sample_width)
        print(f"Yielding WAV header ({len(header)} bytes)...")
        yield header

        # 2. Yield audio chunks from the queue
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                print("Received end signal. Stopping stream.")
                # Optional: Clean up queue task if needed, though get() handles it
                # audio_queue.task_done()
                break  # End of audio data
            # print(f"Yielding audio chunk ({len(chunk)} bytes)...") # Can be noisy
            yield chunk
            all_audio_segments.append(chunk)
            # Mark task as done (important if using queue.join())
            audio_queue.task_done()

        # Optional: Wait for producer task to ensure it finished cleanly,
        # especially if you need to catch exceptions from it here.
        try:
            await producer_task
        except Exception as e:
            print(f"Producer task finished with error: {e}")
            # Handle error if needed (e.g., log it), although stream already ended.

        # Report final performance metrics
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total speech generation completed in {total_time:.2f} seconds")

        # Calculate combined duration
        if all_audio_segments:
            total_bytes = sum(len(segment) for segment in all_audio_segments)
            duration = total_bytes / (2 * SAMPLE_RATE)  # 2 bytes per sample at 24kHz
            print(f"Generated {len(all_audio_segments)} audio segments")
            print(
                f"Generated {duration:.2f} seconds of audio in {total_time:.2f} seconds"
            )
            print(f"Realtime factor: {duration/total_time:.2f}x")

        print(f"Total speech generation completed in {total_time:.2f} seconds")

    # Return the streaming response
    return StreamingResponse(
        stream_generator(),
        media_type="audio/wav",
    )

    # token_gen = generate_tokens_from_api(
    #     prompt=request.input,
    #     voice=request.voice,
    # )

    # samples = tokens_decoder(token_gen)

    # f = io.BytesIO()
    # wav_file = wave.open(f, "wb")
    # wav_file.setnchannels(1)
    # wav_file.setsampwidth(2)
    # wav_file.setframerate(SAMPLE_RATE)

    # async def audio_generator():
    #     try:
    #         write_buffer = bytearray()
    #         buffer_max_size = 1024 * 1024  # 1MB max buffer size (adjustable)

    #         async for audio in samples:
    #             write_buffer.extend(audio)

    #             # Flush buffer if it's large enough
    #             if len(write_buffer) >= buffer_max_size:
    #                 print(f"Flushing buffer: {len(write_buffer)} bytes")
    #                 wav_file.writeframes(write_buffer)
    #                 write_buffer = bytearray()  # Reset buffer

    #         if len(write_buffer) > 0:
    #             print(f"Final buffer flush: {len(write_buffer)} bytes")
    #             wav_file.writeframes(write_buffer)

    #         # Close WAV file if opened
    #         wav_file.close()
    #     except Exception as e:
    #         print(f"⚠️ Error generating audio: {e}")
    #         raise HTTPException(status_code=500, detail=str(e))

    # asyncio.create_task(audio_generator())

    # def iterfile():
    #     yield from f

    # # Return audio file
    # return StreamingResponse(
    #     iterfile(),
    #     media_type="audio/wav",
    # )


if __name__ == "__main__":
    import uvicorn

    # Check for required settings
    required_settings = ["ORPHEUS_HOST", "ORPHEUS_PORT"]
    missing_settings = [s for s in required_settings if s not in os.environ]
    if missing_settings:
        print(f"⚠️ Missing environment variable(s): {', '.join(missing_settings)}")
        print("   Using fallback values for server startup.")

    # Get host and port from environment variables with better error handling
    try:
        host = os.environ.get("ORPHEUS_HOST")
        if not host:
            print("⚠️ ORPHEUS_HOST not set, using 0.0.0.0 as fallback")
            host = "0.0.0.0"
    except Exception:
        print("⚠️ Error reading ORPHEUS_HOST, using 0.0.0.0 as fallback")
        host = "0.0.0.0"

    try:
        port = int(os.environ.get("ORPHEUS_PORT", "5005"))
    except (ValueError, TypeError):
        print("⚠️ Invalid ORPHEUS_PORT value, using 5005 as fallback")
        port = 5005

    print(f"🔥 Starting Orpheus-FASTAPI Server on {host}:{port}")
    print(
        f"💬 Web UI available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
    )
    print(
        f"📖 API docs available at http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs"
    )

    # Read current API_URL for user information
    api_url = os.environ.get("ORPHEUS_API_URL")
    if not api_url:
        print(
            "⚠️ ORPHEUS_API_URL not set. Please configure in .env file before generating speech."
        )
    else:
        print(f"🔗 Using LLM inference server at: {api_url}")

    # Include restart.flag in the reload_dirs to monitor it for changes
    extra_files = ["restart.flag"] if os.path.exists("restart.flag") else []

    # Start with reload enabled to allow automatic restart when restart.flag changes
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
    )
