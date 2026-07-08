FROM python:3.11-slim

WORKDIR /app

# Install the package with the extras the API image needs (server + Qdrant).
COPY pyproject.toml README.md LICENSE ./
COPY atomir ./atomir
RUN pip install --no-cache-dir ".[qdrant,api]"

EXPOSE 8000

# Bind to 0.0.0.0 so the port is reachable from outside the container.
CMD ["uvicorn", "atomir.api:app", "--host", "0.0.0.0", "--port", "8000"]
