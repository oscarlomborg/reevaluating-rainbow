#!/bin/bash
#BSUB -J min_rain_memory
#BSUB -q hpc
#BSUB -n 10
#BSUB -W 02:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -o logs/min_rain_memory_%J.out
#BSUB -e logs/min_rain_memory_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate atarenv

DISABLE_TRACEMALLOC=1 python -u train_memprofile.py \
    --config configs/minatar_rainbow.py \
    --seeds 2 \
    --max-procs 10 \
    --mem-profile \
    --mem-every 5000 \
    --stop-after 200000