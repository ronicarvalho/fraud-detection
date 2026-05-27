#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "manual-$(date +%s)")

# API (Go fraud detector)
docker buildx build \
  --platform linux/amd64 \
  --no-cache \
  -t rpxc/fraud-detector-api:latest \
  -t "rpxc/fraud-detector-api:${COMMIT_SHA}" \
  -f api/Dockerfile \
  --push \
  .

# LB (C fd-passing load balancer)
docker buildx build \
  --platform linux/amd64 \
  --no-cache \
  -t rpxc/fraud-detector-lb:latest \
  -t "rpxc/fraud-detector-lb:${COMMIT_SHA}" \
  -f lb/Dockerfile \
  --push \
  lb/

echo
echo "Pushed:"
echo "  rpxc/fraud-detector-api:latest   /  :${COMMIT_SHA}"
echo "  rpxc/fraud-detector-lb:latest    /  :${COMMIT_SHA}"
echo
echo "Para forçar o avaliador a usar a versão nova:"
echo "  - docker-compose.yml: image: rpxc/fraud-detector-api:${COMMIT_SHA}"
echo "                       image: rpxc/fraud-detector-lb:${COMMIT_SHA}"
echo "  - git commit + push na branch submissão"
