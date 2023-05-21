#!/bin/bash
#
# Runs Redis cache in Docker.

set -e

set -a
source .env
set +a

docker stop "${REDIS_HOST}" &>/dev/null || true

docker run \
  --rm \
  -d \
  --name \
  "${REDIS_HOST}" \
  -p "${REDIS_PORT}:6379" \
  "redis:${REDIS_VERSION}"
