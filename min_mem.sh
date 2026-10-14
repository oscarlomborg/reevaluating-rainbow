#!/bin/bash
#BSUB -J min_rain_memory
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o logs/min_rain_memory_%J.out
#BSUB -e logs/min_rain_memory_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate atarenv

python -u train_memprofile.py \
    --config configs/minatar_rainbow.py \
    --only-env MinAtar/Breakout-v1 \
    --seeds 1 \
    --max-procs 1 \
    --mem-profile \
    --mem-every 5000 \
    --stop-after 200000