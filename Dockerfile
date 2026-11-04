FROM python:3.12-slim

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependencies manifest
COPY pyproject.toml README.md ./

# Install python dependencies using uv
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install -e .

ENV PATH="/opt/venv/bin:$PATH"

# Copy application source code and web static files
COPY app/ ./app/
COPY .env* ./

# Ensure data directories exist for Delta Lake storage
RUN mkdir -p /app/data/delta/model_metrics

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/api/v1/models').raise_for_status()" || exit 1

# Start FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

