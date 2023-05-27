#!/bin/bash
#
# Stops Redis cache in Docker.

set -e

set -a
source .env
set +a

docker stop "${REDIS_HOST}" &>/dev/null || true
docker remove "${REDIS_HOST}" &>/dev/null || true
