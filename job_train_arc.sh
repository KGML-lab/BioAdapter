#!/bin/bash

#SBATCH -J tx+loc_proj
#SBATCH --cpus-per-task=12
#SBATCH --time=40:00:00 
#SBATCH --gres=gpu:4
#SBATCH --partition=h200_normal_q
#SBATCH --account=memtrack
#SBATCH --output=/scratch/bio_diffusion/slurm_logs/%j-%x.out


########## SBATCH --qos=tc_a100_normal_short

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=15897
export WORLD_SIZE2=14


# Threads = cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_NTASKS="${SLURM_NTASKS}
echo "SLURM_PROCID="${SLURM_PROCID}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Load required modules and activate Conda environment
module reset
module load Miniconda3/24.7.1-0
conda init
source ~/.bashrc
source activate bio_up
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


echo "Starting accelerate..."

## context 4

accelerate launch --num_processes 4 --multi_gpu train.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--image_encoder_path="/home/mridul/IP-Adapter-fork/sdxl_models/image_encoder" \
--data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_arc.json" \
--data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
--mixed_precision="fp16" \
--resolution=512 \
--train_batch_size=64 \
--dataloader_num_workers=4 \
--learning_rate=1e-04 \
--weight_decay=0.01 \
--output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/sdxl_extra_context4" \
--save_steps=2000 \
--report_to="wandb" \
--model_type="bioclip" 

