#!/bin/bash
#SBATCH -J biocap_bioclip_inat_birds_minus_species_2gpus_23hrs
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8                 # 4 procs × 8 workers
#SBATCH --time=23:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=imageomicswithanuj
#SBATCH --output=/scratch/bio_diffusion/slurm_logs/%j-%x.out
#SBATCH --qos=tc_h200_normal_short


export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$(( (RANDOM%20000)+30000 ))

module reset
module load Miniconda3/24.7.1-0
source ~/.bashrc
conda activate bio_up
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

echo "Starting accelerate..."



accelerate launch --multi_gpu --num_machines 1 --num_processes 2 \
  tutorial_train.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --image_encoder_path="/home/mridul/IP-Adapter-fork/models/image_encoder" \
    --data_json_file="/projects/ml4science/DATASETS/iNaturalist/train_mini_birds_subset_arc_biocap_text.json" \
    --data_root_path="/projects/ml4science/DATASETS/iNaturalist/images" \
    --mixed_precision="fp16" \
    --resolution=512 \
    --train_batch_size=64 \
    --dataloader_num_workers=8 \
    --learning_rate=1e-04 \
    --weight_decay=0.01 \
    --output_dir="/scratch/bio_diffusion/new_experiments/biocap_taxaadpter/biocap_bioclip_inat_birds_minus_species_2gpus_23hrs" \
    --save_steps=2000 \
    --report_to="wandb" \
    --clip_extra_context_tokens=4 \
    --model_type="bioclip" \

