#!/bin/bash
#SBATCH -J birds_bioclip_4gpus_1.4e-04_all_7levels_COMBINED
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8                 # 4 procs × 8 workers
#SBATCH --time=11:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=imageomicswithanuj
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


accelerate launch --num_machines 1 --num_processes 2 --multi_gpu tutorial_train.py \
--pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
--image_encoder_path="/home/mridul/IP-Adapter-fork/sdxl_models/image_encoder" \
--data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
--data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
--output_dir="/scratch/bio_diffusion/ip-adapter_runs/model_runs/supplementary/SigLIP_birds_subset_old_gpus2_512_24hrs_batch_64" \
--mixed_precision="fp16" \
--resolution=512 \
--train_batch_size=64 \
--dataloader_num_workers=8 \
--learning_rate=1e-04 \
--weight_decay=0.01 \
--save_steps=5000 \
--report_to="wandb" \
--model_type='bioclip'



accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
  tutorial_train_combined.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --image_encoder_path="/home/mridul/IP-Adapter-fork/models/image_encoder" \
    --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_BIRDS_common_name_WITH_COORDS.json" \
    --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
    --mixed_precision="fp16" \
    --resolution=512 \
    --train_batch_size=64 \
    --dataloader_num_workers=8 \
    --learning_rate=1.4e-04 \
    --weight_decay=0.01 \
    --output_dir="/scratch/bio_diffusion/ip-adapter_runs/inat-birds/birds_bioclip_4gpus_1.4e-04_all_7levels_COMBINED" \
    --save_steps=2000 \
    --report_to="wandb" \
    --clip_extra_context_tokens=4 \
    --model_type="bioclip"


#### bioclip baseline ####
# accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#   tutorial_train.py \
#     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#     --image_encoder_path="/home/mridul/IP-Adapter-fork/models/image_encoder" \
#     --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
#     --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
#     --mixed_precision="fp16" \
#     --resolution=512 \
#     --train_batch_size=64 \
#     --dataloader_num_workers=8 \
#     --learning_rate=2e-04 \
#     --weight_decay=0.01 \
#     --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/bioclip_4gpus_2e-04" \
#     --save_steps=2000 \
#     --report_to="wandb" \
#     --clip_extra_context_tokens=4 \
#     --model_type="bioclip"


# ### taxabind + location sequential concat ###
# accelerate launch --multi_gpu --num_machines 1 --num_processes 2 \
#   tutorial_train.py \
#     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#     --image_encoder_path="/home/mridul/IP-Adapter-fork/models/image_encoder" \
#     --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc.json" \
#     --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
#     --mixed_precision="fp16" \
#     --resolution=512 \
#     --train_batch_size=64 \
#     --dataloader_num_workers=8 \
#     --learning_rate=1e-04 \
#     --weight_decay=0.01 \
#     --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/taxa_loc_seq_concat_2gpus" \
#     --save_steps=2000 \
#     --report_to="wandb" \
#     --clip_extra_context_tokens=4 \
#     --model_type="taxa_loc_seq_concat"


#### bioclip with coords ####
# accelerate launch --multi_gpu --num_machines 1 --num_processes 4 \
#   tutorial_train.py \
#     --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
#     --image_encoder_path="/home/mridul/IP-Adapter-fork/models/image_encoder" \
#     --data_json_file="/projects/ml4science/DATASETS/iNaturalist/images/train_reformatted_WITH_COORDS.json" \
#     --data_root_path="/scratch/bio_diffusion/dataset/iNaturalist" \
#     --mixed_precision="fp16" \
#     --resolution=512 \
#     --train_batch_size=64 \
#     --dataloader_num_workers=8 \
#     --learning_rate=1e-04 \
#     --weight_decay=0.01 \
#     --output_dir="/scratch/bio_diffusion/ip-adapter_runs/bioclip/bioclip_4gpus_with_coords" \
#     --save_steps=10000 \
#     --report_to="wandb" \
#     --clip_extra_context_tokens=4 \
#     --image_encoder="bioclip"





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
