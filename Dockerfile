FROM python:3.10-slim

WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies from public PyPI
RUN pip install --no-cache-dir --index-url https://pypi.org/simple .

# Copy all application modules
COPY *.py ./
COPY agent/ ./agent/
COPY schemas/ ./schemas/

CMD ["python", "main.py"]
