#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${PYTHON_BIN:-$(command -v python3)}"
mkdir -p "$root/output"

for seed in 1 2 3; do
    printf -v command '%q ' "$python" -u "$root/run.py" \
        --data-path="${CAR_DATA:-/u/mlowery/dgpo/datasets/car.npz}" \
        "$@" --seed="$seed" --device=cuda --wandb-mode=online
    sbatch --job-name="car_dse_seed${seed}" \
        --chdir="$root" \
        --output="$root/output/%x_%j.out" \
        --error="$root/output/%x_%j.err" <<EOF
#!/usr/bin/env bash
#SBATCH --partition=gpuA100x4
#SBATCH --account=bgcs-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
export OMP_NUM_THREADS=4
$command
EOF
done
