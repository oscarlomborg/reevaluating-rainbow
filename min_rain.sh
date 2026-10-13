#!/bin/bash
#BSUB -J min_rain
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 01:00
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=500MB]"
#BSUB -o logs/moe_hpc_%J.out
#BSUB -e logs/moe_hpc_%J.err


source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate atarenv

python -u train.py --config configs\minatar_rainbow.py --max-procs 8 --seeds 4