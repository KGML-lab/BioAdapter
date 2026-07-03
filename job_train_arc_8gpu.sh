#!/bin/bash
#SBATCH -J bioclip_8gpu3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1                # One task per node
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24                 # All CPUs for this node
#SBATCH --time=2-00:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --account=imageomicswithanuj
#SBATCH --output=/scratch/bio_diffusion/slurm_logs/%j-%x.out

# Good hygiene
export OMP_NUM_THREADS=1

# NCCL Configuration for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_ASYNC_ERROR_HANDLING=1

# Rendezvous (same for all ranks)
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=12341

# Env setup
module reset
module load Miniconda3/24.7.1-0
source ~/.bashrc
conda activate bio_up
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

echo "NODELIST: $SLURM_NODELIST"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Use accelerate launch - one launcher per node, each spawns 4 GPU processes
srun accelerate launch \
  --num_machines=$SLURM_NNODES \
  --num_processes=8 \
  --machine_rank=$SLURM_PROCID \
  --main_process_ip=$MASTER_ADDR \
  --main_process_port=$MASTER_PORT \
  tutorial_train.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --image_encoder_path="/home/mridul/IP-Adapter-fork/models/image_encoder" \
  --data_json_file="/projects/ml4science/DATASETS/iNaturalist/images/train_reformatted_WITH_COORDS.json" \
  --data_root_path="/scratch/bio_diffusion/dataset/iNaturalist" \
  --resolution=512 \
  --mixed_precision="fp16" \
  --train_batch_size=64 \
  --dataloader_num_workers=6 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/bioclip_8gpus_with_coords_a100" \
  --save_steps=10000 \
  --report_to="wandb" \
  --clip_extra_context_tokens=4 \
  --image_encoder="bioclip"
