# CUDA training image for forex-trainer.
# Build context must be the PARENT directory holding both repositories
# (see compose.yaml), because forex-env-v3 is a path dependency (ADR-0001).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# uv manages its own Python; no system python needed.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python

WORKDIR /work/forex-trainer

# Dependency layers first for build caching.
COPY forex-env-v3/ /work/forex-env-v3/
COPY forex-trainer/pyproject.toml forex-trainer/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Project source last (changes most often).
COPY forex-trainer/ /work/forex-trainer/
RUN uv sync --frozen --no-dev

ENTRYPOINT []
CMD ["uv", "run", "forex-train", "--help"]
