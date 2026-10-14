#!/bin/bash
# Create logs/ before submitting: mkdir -p logs
#BSUB -J "min_noisy[1-5]%5"
#BSUB -q hpc
#BSUB -n 10
#BSUB -W 30:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -o logs/min_noisy_%J_%I.out
#BSUB -e logs/min_noisy_%J_%I.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate atarenv

# train.py performs periodic glibc trimming in each seed worker.
export MALLOC_TRIM_EVERY=1000

python -u train_array.py \
    --config configs/minatar_noisy.py \
    --seeds 10 \
    --seed-procs 10
