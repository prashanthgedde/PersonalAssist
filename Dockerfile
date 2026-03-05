FROM python:3.10-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using public PyPI (uv.lock may reference internal Netflix PyPI)
RUN UV_INDEX_URL=https://pypi.org/simple uv sync --frozen --no-dev

# Copy application code
COPY main.py tools.py memory.py ./

# Persistent volume mount point for chroma_db
RUN mkdir -p /data/chroma_db

CMD ["uv", "run", "python", "main.py"]
