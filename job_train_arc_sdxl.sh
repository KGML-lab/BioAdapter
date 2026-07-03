#!/bin/bash
#SBATCH -J sdxl_BioCLIP_birds_subset_old_gpus4_1024_24hours
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8                 # 4 procs × 8 workers
#SBATCH --time=24:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=ml4science
#SBATCH --qos=tc_h200_normal_short
#SBATCH --output=/scratch/bio_diffusion/slurm_logs/%j-%x.out


export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$(( (RANDOM%20000)+30000 ))

module reset
module load Miniconda3/24.7.1-0
source ~/.bashrc
conda activate bio_up
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

echo "Starting accelerate..."


accelerate launch --num_processes 4 --multi_gpu tutorial_train_sdxl.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
--image_encoder_path="/home/mridul/IP-Adapter-fork/sdxl_models/image_encoder" \
--data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
--output_dir="/scratch/bio_diffusion/ip-adapter_runs/model_runs/supplementary/sdxl_BioCLIP_birds_subset_old_gpus2_1024_24hrs_batch_12" \
--mixed_precision="fp16" \
--resolution=1024 \
--train_batch_size=12 \
--dataloader_num_workers=8 \
--learning_rate=1e-04 \
--weight_decay=0.01 \
--save_steps=5000 \
--report_to="wandb" \
--model_type='bioclip'

