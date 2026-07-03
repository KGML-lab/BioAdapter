#!/bin/bash

#SBATCH -J tx+loc_proj
#SBATCH --cpus-per-task=12
#SBATCH --time=11:00:00 
#SBATCH --gres=gpu:4
#SBATCH --partition=h200_normal_q
#SBATCH --account=memtrack
#SBATCH --output=/scratch/bio_diffusion/slurm_logs/%j-%x.out


#### SBATCH --qos=tc_a100_normal_short

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

# accelerate launch --num_processes 4 --multi_gpu tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/mridul/IP-Adapter-fork/sdxl_models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/sdxl_extra_context4" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=4 

## context 1

# accelerate launch --num_processes 4 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/sdxl_models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/sdxl_extra_context1" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=1 

### no projection layer


# accelerate launch --num_processes 2 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/no_proj" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=1 \
# --no_projection_layer






#########################
# asic ip adapter debug run
# accelerate launch --num_processes 4 --multi_gpu tutorial_train.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter/models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/debug_save" \
# --save_steps=2000 \
# --report_to="wandb" 








##############################################

### combined   bioclip + location adapter runs
# accelerate launch --num_processes 4 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train_location.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/bioclip_location_context4" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=4 \
# --image_encoder='bioclip' \
# --image_encoder_secondary='location' \
# --image_proj_secondary 


##### combined  taxabind + location double projection layers adapter runs 
accelerate launch --num_processes 4 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train_location.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/models/image_encoder" \
--data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
--mixed_precision="fp16" \
--resolution=512 \
--train_batch_size=64 \
--dataloader_num_workers=4 \
--learning_rate=1e-04 \
--weight_decay=0.01 \
--output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/taxabind_location_context4_doubleproj" \
--save_steps=2000 \
--report_to="wandb" \
--clip_extra_context_tokens=4 \
--image_encoder='taxabind' \
--image_encoder_secondary='location' \
--image_proj_secondary 

##### combined  taxabind + location single projection layer conct embeds adapter runs
# accelerate launch --num_processes 4 --multi_gpu --main_process_port ${MASTER_PORT} tutorial_train_location.py \
# --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
# --image_encoder_path="/home/medha/Projects/bio-diffusion/IP-Adapter-fork/models/image_encoder" \
# --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
# --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
# --mixed_precision="fp16" \
# --resolution=512 \
# --train_batch_size=64 \
# --dataloader_num_workers=4 \
# --learning_rate=1e-04 \
# --weight_decay=0.01 \
# --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/taxabind_location_context4_concat" \
# --save_steps=2000 \
# --report_to="wandb" \
# --clip_extra_context_tokens=4 \
# --image_encoder='taxabind' \
# --image_encoder_secondary='location' 