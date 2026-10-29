#!/bin/bash -l
#SBATCH --job-name=opendust-jax-2d
#SBATCH --account=proj_1827
#SBATCH --partition=test
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate opendust-jax

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
# Leave enough GPU memory for the OpenMM CUDA context in the same process.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
python --version
nvidia-smi

srun python -u performance/JAX/benchmark_pair_forces.py \
  --powers 15 16 17 18 \
  --block-sizes 256 \
  --source-block-sizes 128 256 512 1024 \
  --warmup 3 \
  --repeat 10
