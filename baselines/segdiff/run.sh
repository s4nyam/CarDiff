#!/bin/bash
#SBATCH --job-name=sgdddim
#SBATCH --account=project_465001696
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=1

srun singularity exec -B /scratch/project_465001696/playground/0.3.segguideddiff py_cotainr.sif python main.py --model_type DDIM --resume_epoch 880 --segmentation_guided 