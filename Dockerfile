FROM ghcr.io/astral-sh/uv:0.6.14-python3.13-bookworm

# Set non-interactive frontend to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal requirements for uv
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    # portaudio19-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Create non-root user and set up directories
RUN useradd -m -u 1001 appuser && \
mkdir -p /app && \
chown -R appuser:appuser /app

# Switch to non-root user
USER appuser
WORKDIR /app

# Copy dependency files
COPY --chown=appuser:appuser pyproject.toml uv.lock /app/
    
# Use uv to create virtual environment, install Python and dependencies
RUN uv sync

# Install PyTorch with CUDA support and other dependencies
# RUN uv pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy source code and configuration
COPY --chown=appuser:appuser src /app/src
# COPY --chown=appuser:appuser .env.example /app/

# Create .env file from example if it doesn't exist
# RUN cp -n .env.example .env || true

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    USE_GPU=true
    # PATH="/app/.venv/bin:$PATH"

# Expose the port
EXPOSE 8000

# Start the server using uv run as specified
CMD ["uv", "run", "src/server.py"]
