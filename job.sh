#!/bin/bash
#SBATCH --job-name=ip_adapter
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --partition=a100

# ============================================================
# Single-Node Multi-GPU DPO Training with SLURM
# ============================================================
# This script launches DPO training on a single node with multiple GPUs.
# Uses torchrun for distributed training within the node.
# ============================================================

echo "============================================"
echo "DPO Training Job Started (Single Node)"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Working directory: $(pwd)"
echo "============================================"


# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=0

# Get number of GPUs
NGPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NGPUS GPUs"

source ~/miniconda3/etc/profile.d/conda.sh

conda activate apadapter


python inference.py --num_samples 10 --model_type bioclip \
--ip_ckpt /home/karimimonsefi.1/BioAdapter/ip_adapter.bin \
--json /home/karimimonsefi.1/BioAdapter/subset_inat_200_mapped.json \
--out_dir /home/karimimonsefi.1/BioAdapter/output

echo "============================================"
echo "DPO Training Job Completed"
echo "============================================"
