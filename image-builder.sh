#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

docker buildx build \
  --platform linux/amd64 \
  -t rpxc/fraud-detector-api:latest \
  -f api/Dockerfile \
  --push \
  .
