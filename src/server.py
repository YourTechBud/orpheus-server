import os
import time
from enum import Enum

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from inference import (
    DEFAULT_VOICE,
    ffmpeg_opus_stream_generator,
    generate_tokens_from_api,
    tokens_decoder_sync,
)


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


# Enum for response formats
class ResponseFormatEnum(str, Enum):
    WAV = "wav"
    OPUS = "opus"


# API models
class SpeechRequest(BaseModel):
    input: str
    model: str = "orpheus"
    voice: str = DEFAULT_VOICE
    response_format: ResponseFormatEnum = ResponseFormatEnum.WAV
    speed: float = 1.0


class Model(BaseModel):
    id: str
    created: int
    object: str = "model"
    owned_by: str


class ListModelsResponse(BaseModel):
    object: str = "list"
    data: list[Model]


# Get process start time in seconds since epoch
process_start_time = int(time.time())


@app.get("/v1/models")
async def list_models():
    model_name = os.environ.get("ORPHEUS_MODEL_NAME", "Orpheus-3b-FT-Q8_0.gguf")
    return ListModelsResponse(
        data=[Model(id=model_name, created=process_start_time, owned_by="inferix")]
    )


# OpenAI-compatible API endpoint
@app.post("/v1/audio/speech")
async def create_speech_api(request: SpeechRequest):
    """
    Generate speech from text using the Orpheus TTS model.
    Compatible with OpenAI's /v1/audio/speech endpoint.
    Supports WAV and Opus streaming output.
    """
    if not request.input:
        raise HTTPException(status_code=400, detail="Missing input text")

    # Generate audio using the synchronous generator from inference.py
    wav_audio_generator = tokens_decoder_sync(
        generate_tokens_from_api(
            prompt=request.input,
            voice=request.voice,
        ),
        # output_file=None is the default, meaning it yields chunks
    )

    if request.response_format == ResponseFormatEnum.OPUS:
        # print("Streaming Opus audio...")
        return StreamingResponse(
            ffmpeg_opus_stream_generator(wav_audio_generator), media_type="audio/opus"
        )
    elif request.response_format == ResponseFormatEnum.WAV:
        # print("Streaming WAV audio...")
        return StreamingResponse(
            wav_audio_generator,  # This generator already includes the WAV header
            media_type="audio/wav",
        )
    else:
        # Should not happen due to Pydantic validation with Enum
        raise HTTPException(status_code=400, detail="Invalid response format")


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
        workers=int(os.environ.get("NUM_WORKERS", 1)),
    )
