FROM python:3.10-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies from public PyPI
RUN pip install --no-cache-dir --index-url https://pypi.org/simple .

# Copy application code
COPY main.py tools.py memory.py ./

# Persistent volume mount point for chroma_db
RUN mkdir -p /data/chroma_db

CMD ["python", "main.py"]
