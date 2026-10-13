#!/bin/bash
#BSUB -J min_rain_test
#BSUB -q hpc
#BSUB -n 10
#BSUB -W 06:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"
#BSUB -o logs/min_rain_test_%J.out
#BSUB -e logs/min_rain_test_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate atarenv

python -u train.py --config configs/minatar_rainbow.py --max-procs 10 --seeds 2