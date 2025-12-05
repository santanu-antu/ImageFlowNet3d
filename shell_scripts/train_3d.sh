#!/bin/bash

#SBATCH --job-name=train_3d_imageflownet
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --mail-type=ALL
#SBATCH --output=../sbatch_logs/train_3d_%j.out
#SBATCH --error=../sbatch_logs/train_3d_%j.err


set -euo pipefail
mkdir -p ../sbatch_logs

module --force purge
module load StdEnv
module load miniconda

# Activate conda environment
set +u
source /apps/software/2022b/software/miniconda/24.11.3/etc/profile.d/conda.sh
conda activate /nfs/roberts/project/pi_sk2433/sa2556/.conda/envs/imageflownet
set -u

cd /nfs/roberts/project/pi_sk2433/sa2556/ImageFlowNet

# Diagnostics
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__, "built cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
PY

# Training 3D ImageFlowNetODE on synthetic 3D data
# Volume size: 64x64x64
# ODE location: all_connections (9 ODE blocks)
# Attention: only at resolution 8 for memory efficiency
python src/train_3d.py \
  --mode train \
  --max-epochs 120 \
  --dataset-name synthetic3d \
  --dataset-path /nfs/roberts/project/pi_sk2433/sa2556/ImageFlowNet/data/synthesized3d \
  --image-folder base \
  --volume-size 64 \
  --ode-location all_connections \
  --attention-resolutions "16,8" \
  --num-filters 64 \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --max-training-samples 512 \
  --max-validation-samples 64 \
  --coeff-smoothness 0.0 \
  --coeff-contrastive 0.0 \
  --gpu-id 0
