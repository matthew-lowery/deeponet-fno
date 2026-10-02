#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for seed in 1 2 3; do
    python3 -u run.py "$@" --seed="$seed"
done
