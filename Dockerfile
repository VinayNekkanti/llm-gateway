# Two stages: build the virtual environment with uv, then copy only what's needed to run
# into a small image (no uv, no build caches, no compilers).

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first: this layer is reused until uv.lock changes, so code edits rebuild fast
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Download the embedding model at build time, so containers start without internet access
ENV FASTEMBED_CACHE_PATH=/app/models
RUN /app/.venv/bin/python -c \
    "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"


FROM python:3.12-slim
# Run as a normal user, not root
RUN useradd --create-home --uid 1000 gateway
WORKDIR /app
COPY --from=builder --chown=gateway /app /app
COPY --chown=gateway config.yaml ./config.yaml

ENV PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/models \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000
USER gateway
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["llm-gateway"]
