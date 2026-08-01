#!/bin/bash

sp() {
    local pycmd="$1"
    local hr="${2:-8}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA100x8,gpuA100x4,gpuH200x8
#SBATCH --account=bgcs-delta-gpu
#SBATCH --job-name=fno_laplace
#SBATCH --time=${hr}:00:00
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --output=./output/%x_%A.out
#SBATCH --error=./output/%x_%A.err

module purge
export PATH=/u/mlowery/.conda/envs/jax_eqx/bin:\$PATH
cd /u/mlowery/deeponet-fno/src/
$pycmd
EOF
}

mkdir -p output
project=laplace_fno
data=/u/mlowery/dgpo/datasets
epochs=500
ee=50
samples=50
hbatch=10
hprobes=4
prior=100.0
maxstd=1000000.0
init_noise=1.0
damping=1e-6

depth=4; width=64; modes=16; proj=128; batch=20; tbatch=100; lr=0.001
sp "python -u fno_laplace.py --dataset=burgers --data-dir=$data --epochs=$epochs --eval-every=$ee --batch-size=$batch --test-batch-size=$tbatch --lr=$lr --depth=$depth --width=$width --modes=$modes --proj-dim=$proj --laplace-samples=$samples --hessian-batches=$hbatch --hessian-probes=$hprobes --prior-precision=$prior --likelihood-noise=$init_noise --laplace-damping=$damping --laplace-max-std=$maxstd --wandb --wandb-project=$project --name=burgers_fno_laplace" 8

depth=4; width=32; modes=12; proj=128; batch=20; tbatch=100; lr=0.001
sp "python -u fno_laplace.py --dataset=darcy --data-dir=$data --epochs=$epochs --eval-every=$ee --batch-size=$batch --test-batch-size=$tbatch --lr=$lr --depth=$depth --width=$width --modes=$modes --proj-dim=$proj --laplace-samples=$samples --hessian-batches=$hbatch --hessian-probes=$hprobes --prior-precision=$prior --likelihood-noise=$init_noise --laplace-damping=$damping --laplace-max-std=$maxstd --wandb --wandb-project=$project --name=darcy_fno_laplace" 8

depth=4; width=64; modes=32; proj=128; batch=20; tbatch=100; lr=0.001
sp "python -u fno_laplace.py --dataset=beijing --data-dir=$data --epochs=$epochs --eval-every=$ee --batch-size=$batch --test-batch-size=$tbatch --lr=$lr --depth=$depth --width=$width --modes=$modes --proj-dim=$proj --laplace-samples=$samples --hessian-batches=$hbatch --hessian-probes=$hprobes --prior-precision=$prior --likelihood-noise=$init_noise --laplace-damping=$damping --laplace-max-std=$maxstd --wandb --wandb-project=$project --name=beijing_fno_laplace" 8
