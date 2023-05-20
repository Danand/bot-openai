#!/bin/bash
#
# Runs Redis cache in Docker.

set -e

set -a
source .env
set +a

container_name="bot-openai-redis"

docker stop "${container_name}" &>/dev/null || true

docker run \
  --rm \
  -d \
  --name \
  "${container_name}" \
  -p  "${REDIS_PORT}:6379" \
  "redis:${REDIS_VERSION}"
