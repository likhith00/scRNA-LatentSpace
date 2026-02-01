#!/bin/bash
#SBATCH --job-name=RNAfirstrunv2

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=00:20:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

source .venv/bin/activate

python src/eval.py --params "params.yaml" --dataset "digits" --run-dir outputs/run_fae4cd8