#!/bin/bash
#SBATCH --job-name=syn_seg
#SBATCH --account=project_465002351
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=1

# ================================
# Create job-specific MIOpen cache
# ================================

export MIOPEN_USER_DB_PATH=/scratch/project_465002351/$USER/miopen_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

echo "Using MIOpen cache at: $MIOPEN_USER_DB_PATH"

# ================================
# Run inside Singularity
# ================================

srun singularity exec \
    --env MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH \
    --env MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH \
    -B /scratch/project_465002351/cardiff/ \
    -B /scratch/project_465002351/cardiff-seg/ \
    -B $MIOPEN_USER_DB_PATH:$MIOPEN_USER_DB_PATH \
    py_cotainr.sif \
    python fill_mask.py \
    --masks_dir /scratch/project_465002351/cardiff-seg/all_masks/batch_000/ \
    --output_dir /scratch/project_465002351/cardiff-seg/all_syns/batch_000