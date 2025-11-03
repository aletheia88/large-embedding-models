#!/bin/bash
#SBATCH --job-name=re
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=120g
#SBATCH --requeue
#SBATCH --output=slurm-%j.out

MICROMAMBA_BIN=${MICROMAMBA_BIN:-/orcd/scratch/orcd/010/jianggy/micromamba/bin/micromamba}
if [[ ! -x $MICROMAMBA_BIN ]]; then
    log "micromamba binary missing at $MICROMAMBA_BIN"
    exit 1
fi

source <("$MICROMAMBA_BIN" shell hook --shell=bash)


cd /orcd/scratch/orcd/010/jianggy/smt/RAE
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=src micromamba run -n rae torchrun --standalone --nproc_per_node=2 -m feature_mae.train_feature_mae \
    --data-path /orcd/scratch/bcs/002/jianggy/imagenet1k_wds \
    --epochs 800 \
    --steps-per-epoch 2503 \
    --steps-are-global \
    --batch-size 512 \
    --accumulation-steps 4 \
    --mask-ratio 0.75 \
    --precision bf16 \
    --save-interval 20 \
    --num-workers 12 \
    --include-special \
    --output-dir results/feature_mae_vitb_accum_alltokens \
    --resume /orcd/scratch/orcd/010/jianggy/smt/RAE/results/feature_mae_vitb_accum_alltokens/checkpoint_400.pth
