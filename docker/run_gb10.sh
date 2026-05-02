#!/usr/bin/env bash
# run_gb10.sh — launch a training run on the GB10 using docker compose
# Usage:  ./docker/run_gb10.sh [config] [extra args...]
#   e.g.  ./docker/run_gb10.sh configs/gb10_large.yaml
#         ./docker/run_gb10.sh configs/gb10_large.yaml --episodes 100000
#
# Runs as root inside the container (required for CUDA/JAX on some systems).
# After the run, fix ownership with:  ./docker/run_gb10.sh --fix-perms

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# --fix-perms: chown runs/ back to the current user
if [ "${1:-}" = "--fix-perms" ]; then
    docker run --rm \
        -v "$REPO_DIR/runs:/app/runs" \
        alpine \
        chown -R "$(id -u):$(id -g)" /app/runs
    echo "✓ Ownership of runs/ fixed to $(id -u):$(id -g)"
    exit 0
fi

CONFIG="${1:-configs/gb10_large.yaml}"
shift || true   # remaining args forwarded to the training command

mkdir -p runs

echo "▶ Launching training on GB10"
echo "  Config : $CONFIG"
echo "  Runs   : $REPO_DIR/runs/"
echo

docker compose run \
    --name training_run \
    -d \
    gb10 \
    poset_rl.train --config "$CONFIG" "$@"

echo
echo "Container started.  Monitor with:"
echo "  docker logs -f training_run"
echo
echo "When training finishes, fix file ownership with:"
echo "  ./docker/run_gb10.sh --fix-perms"
