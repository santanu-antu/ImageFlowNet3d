#!/bin/bash

#SBATCH --job-name=train_retina_ucsf
#SBATCH --partition=pi_krishnaswamy
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4-00:00:00
#SBATCH --mem=32G
#SBATCH --mail-type=ALL
#SBATCH --output=../sbatch_logs/train_retina_ucsf_%j.out
#SBATCH --error=../sbatch_logs/train_retina_ucsf_%j.err

set -euo pipefail
mkdir -p ../sbatch_logs

module --force purge
module load StdEnv
## Avoid relying on site module for conda; we source conda.sh directly below.

# Activate conda environment
# Note: conda activation scripts may reference unset vars; temporarily disable 'nounset'.
set +u
source /vast/palmer/apps/avx2/software/miniconda/24.7.1/etc/profile.d/conda.sh
conda activate imageflownet_gpu
set -u

# Change directory to repo root (use absolute path so SLURM spool doesn’t break it)
REPO_ROOT="/gpfs/gibbs/project/krishnaswamy_smita/sa2556/imageflownet/ImageFlowNet"
cd "$REPO_ROOT"

# Diagnostics
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
echo "PWD: $(pwd)"
echo "which python: $(which python)"
nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__, "built cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

# Train on Retina UCSF
python "$REPO_ROOT/src/train_2pt_all.py" \
  --mode train \
  --dataset-name retina_ucsf \
  --max-epochs 120 \
  --gpu-id 0 \
  --segmentor-ckpt '$ROOT/checkpoints/segment_retinaUCSF_seed1.pty'
