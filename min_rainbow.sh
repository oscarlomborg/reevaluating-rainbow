#!/bin/bash
#BSUB -J min_rain_trim
#BSUB -q hpc
#BSUB -n 10
#BSUB -W 01:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -o logs/min_rain_trim_%J.out
#BSUB -e logs/min_rain_trim_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate atarenv

# train.py performs periodic glibc trimming in each seed worker.
export MALLOC_TRIM_EVERY=1000

python -u train.py --config configs/minatar_rainbow.py --max-procs 10 --seeds 2