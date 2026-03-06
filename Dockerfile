FROM python:3.10-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies from public PyPI
RUN pip install --no-cache-dir --index-url https://pypi.org/simple .

# Copy all application modules
COPY *.py ./

# Ensure mount point exists for local runs (fly.io volume handles this in production)
RUN mkdir -p /data/chroma_db

CMD ["python", "main.py"]
