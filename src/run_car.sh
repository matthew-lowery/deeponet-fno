#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python="${PYTHON_BIN:-/u/mlowery/.conda/envs/gnot/bin/python3}"

for seed in 1 2 3; do
    "$python" -u run.py "$@" --seed="$seed"
done
