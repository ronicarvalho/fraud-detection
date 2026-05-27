#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Tag with both :latest (for convenience) and :<short-commit> (for cache busting on the evaluator).
COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "manual-$(date +%s)")

docker buildx build \
  --platform linux/amd64 \
  --no-cache \
  -t "rpxc/fraud-detector-api:${COMMIT_SHA}" \
  -f api/Dockerfile \
  --push \
  .

echo
echo "Pushed:"
echo "  rpxc/fraud-detector-api:${COMMIT_SHA}"
echo
echo "Para forçar o avaliador a usar a versão nova, no repo da branch submissão:"
echo "  - Edite docker-compose.yml: image: rpxc/fraud-detector-api:${COMMIT_SHA}"
echo "  - git add docker-compose.yml && git commit -m 'submission ${COMMIT_SHA}' && git push"
