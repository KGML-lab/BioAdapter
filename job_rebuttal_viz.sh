#!/bin/bash
#SBATCH --job-name=rebuttal_viz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=a100

# ============================================================
# CVPR Rebuttal - Cross-Attention Visualization
# Demonstrates that taxonomy tokens encode visual traits
# ============================================================

echo "============================================"
echo "CVPR Rebuttal Attention Visualization"
echo "============================================"

export PYTHONUNBUFFERED=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate apadapter

python inference_attention_viz_final.py \
    --ip_ckpt /home/karimimonsefi.1/BioAdapter/ip_adapter.bin \
    --json_file /home/karimimonsefi.1/BioAdapter/subset_inat_200_mapped.json \
    --out_dir /home/karimimonsefi.1/BioAdapter/attention_rebuttal_final \
    --num_samples 200 \
    --model_type bioclip \
    --steps 50 \
    --guidance_scale 7.5 \
    --scale 1.0 \
    --seed 42

echo "============================================"
echo "Done! Check attention_rebuttal/ for figures"
echo "============================================"
