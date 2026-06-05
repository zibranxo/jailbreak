FROM python:3.11-slim

# Set up user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user:user . .

# Create persistent directories for SQLite DB and logs and ensure permissions
RUN mkdir -p data/ logs/ && \
    chown -R user:user data/ logs/ && \
    chmod -R 777 data/ logs/

# Switch to the non-root user
USER user

# Download models during build so it's cached in the Docker image
RUN python scripts/download_models.py

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Start FastAPI on port 7860
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]