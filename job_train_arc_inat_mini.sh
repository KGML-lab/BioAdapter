#!/bin/bash
#SBATCH -J bioclip_inat_mini_2gpus_126hrs
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8                 # 4 procs × 8 workers
#SBATCH --time=126:30:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=imageomicswithanuj
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


### taxabind + location sequential concat ###
# accelerate launch --multi_gpu --num_machines 1 --num_processes 2 \
#   tutorial_train.py \
#     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#     --image_encoder_path="/home/mridul/IP-Adapter-fork/models/image_encoder" \
#     --data_json_file="/projects/ml4science/DATASETS/iNaturalist/images/train_mini_reformatted_WITH_COORDS.json" \
#     --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
#     --mixed_precision="fp16" \
#     --resolution=512 \
#     --train_batch_size=64 \
#     --dataloader_num_workers=8 \
#     --learning_rate=1e-04 \
#     --weight_decay=0.01 \
#     --output_dir="/scratch/bio_diffusion/ip-adapter_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs" \
#     --save_steps=10000 \
#     --report_to="wandb" \
#     --clip_extra_context_tokens=4 \
#     --model_type="bioclip"

accelerate launch --multi_gpu --num_machines 1 --num_processes 2 \
  tutorial_train.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --image_encoder_path="/home/mridul/IP-Adapter-fork/models/image_encoder" \
    --data_json_file="/projects/ml4science/DATASETS/iNaturalist/images/train_mini_reformatted_WITH_COORDS.json" \
    --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
    --mixed_precision="fp16" \
    --resolution=512 \
    --train_batch_size=64 \
    --dataloader_num_workers=8 \
    --learning_rate=1e-04 \
    --weight_decay=0.01 \
    --output_dir="/scratch/bio_diffusion/ip-adapter_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs_restart" \
    --save_steps=1000 \
    --report_to="wandb" \
    --clip_extra_context_tokens=4 \
    --model_type="bioclip" \
    --pretrained_ip_adapter_path="/scratch/bio_diffusion/ip-adapter_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-340000"

