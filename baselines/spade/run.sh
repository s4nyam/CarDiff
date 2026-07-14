#!/bin/bash
#SBATCH --job-name=spade
#SBATCH --account=project_465001696
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=1

srun singularity exec -B /scratch/project_465001696/playground/0.0.spade/ py_cotainr.sif python train_multi.py