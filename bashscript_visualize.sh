#!/bin/bash
#SBATCH --job-name=RNAfirstrunv2

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=00:60:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

python src/visualize.py --run-dir outputs/run_a60de80 --umap