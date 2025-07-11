FROM python:3.12-slim-bookworm

ARG GITHUB_REPO=helehh/FastChat
ARG GITHUB_BRANCH=main

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PUID=1000 \
    PGID=1000


RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        git ffmpeg libsm6 libxext6 \
        curl util-linux; \
    rm -rf /var/lib/apt/lists/*


WORKDIR /tmp
RUN set -eux; \
    git clone --depth 1 --branch "${GITHUB_BRANCH}" \
        "https://github.com/${GITHUB_REPO}.git" fastchat; \
    cd fastchat; \
    # install cpu-only version of torch as we're not using it to serve models
    pip install torch --index-url https://download.pytorch.org/whl/cpu; \
    pip install --upgrade ".[model_worker,webui]"; \
    cd ..; \
    rm -rf fastchat


RUN set -eux; \
cat <<'EOF' > /entrypoint.sh
#!/usr/bin/env bash
set -e
exec "$@"
EOF
RUN chmod +x /entrypoint.sh


RUN mkdir -p /workspace
WORKDIR /workspace


ENTRYPOINT ["/entrypoint.sh"]
CMD ["python"]
