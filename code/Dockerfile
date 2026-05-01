# Use the slim Python 3.12 image for a small footprint
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for compiling C++ libraries (ChromaDB uses this)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy existing requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download HuggingFace local models during the build phase so they boot up instantly!
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the entire codebase into the container
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Start the Enterprise FastAPI Server
CMD ["uvicorn", "enterprise.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
