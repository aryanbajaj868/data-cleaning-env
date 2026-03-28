FROM python:3.11-slim

# Create a non-root user for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Switch to root temporarily to install system dependencies
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Switch back to the non-root user
USER user

# Copy and install Python dependencies first (layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=user . .

# HF Spaces convention: expose port 7860
EXPOSE 7860

# Health check (Added '|| exit 1' so Docker registers failures correctly)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Launch the FastAPI server
CMD ["python", "-m", "server.app"]