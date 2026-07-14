#!/bin/bash
#SBATCH --job-name=seg_models
#SBATCH --account=project_465002351
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=small-g
#SBATCH --gpus-per-task=1
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# =====================================
# Define Models
# =====================================

MODELS=("Unet" "UnetPlusPlus" "MAnet" "Linknet" "FPN" "PSPNet" "PAN")

NUM_MODELS=${#MODELS[@]}

# Safety check
if [ "$SLURM_ARRAY_TASK_ID" -ge "$NUM_MODELS" ]; then
    echo "Invalid task ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

MODEL_NAME=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL_NAME"

# Optional small stagger
sleep 30

# =====================================
# Create job-specific MIOpen cache
# =====================================

export MIOPEN_USER_DB_PATH=/scratch/project_465002351/$USER/miopen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

echo "Using MIOpen cache at: $MIOPEN_USER_DB_PATH"

# =====================================
# Run inside Singularity
# =====================================

srun singularity exec \
    --env MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH \
    --env MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH \
    -B /scratch/project_465002351/cardiff-seg/ \
    -B $MIOPEN_USER_DB_PATH:$MIOPEN_USER_DB_PATH \
    py_cotainr.sif \
    python train_seg.py \
        --model $MODEL_NAME