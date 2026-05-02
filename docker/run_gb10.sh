#!/usr/bin/env bash
# run_gb10.sh — launch a training run on the GB10 using docker compose
# Usage:  ./docker/run_gb10.sh [config] [extra args...]
#   e.g.  ./docker/run_gb10.sh configs/gb10_large.yaml
#         ./docker/run_gb10.sh configs/gb10_large.yaml --episodes 100000
#
# The script:
#   1. Exports UID/GID so compose mounts files as the current user (no root-owned CSVs)
#   2. Creates ./runs/ if it doesn't exist
#   3. Runs docker compose in detached mode and tails the logs

set -euo pipefail

CONFIG="${1:-configs/gb10_large.yaml}"
shift || true   # remaining args forwarded to the training command

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p runs

export UID="$(id -u)"
export GID="$(id -g)"

echo "▶ Launching training on GB10"
echo "  Config : $CONFIG"
echo "  UID/GID: $UID/$GID"
echo "  Runs   : $REPO_DIR/runs/"
echo

docker compose run \
  --name training_run \
  --rm \
  -d \
  gb10 \
  python -m poset_rl.train --config "$CONFIG" "$@"

echo
echo "Container started.  Tail logs with:"
echo "  docker logs -f training_run"
