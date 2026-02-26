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
#SBATCH --array=0-130

# ================================
# USER-CONTROLLED DEBUG RANGE
# ================================

START=0
END=130

# Stop if outside desired debug range
if [ "$SLURM_ARRAY_TASK_ID" -lt "$START" ] || [ "$SLURM_ARRAY_TASK_ID" -gt "$END" ]; then
    echo "Task $SLURM_ARRAY_TASK_ID outside range $START-$END. Exiting."
    exit 0
fi

# ================================
# Format batch number (000, 001, ...)
# ================================

BATCH=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})

echo "Processing batch_$BATCH"

# ================================
# Optional 1-minute pause
# ================================

sleep 60

# ================================
# Create job-specific MIOpen cache
# ================================

export MIOPEN_USER_DB_PATH=/scratch/project_465002351/$USER/miopen_${SLURM_JOB_ID}_${BATCH}
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
    --masks_dir /scratch/project_465002351/cardiff-seg/all_masks/batch_${BATCH}/ \
    --output_dir /scratch/project_465002351/cardiff-seg/all_syns/batch_${BATCH}/