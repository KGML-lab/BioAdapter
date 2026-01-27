#!/bin/bash
#SBATCH --job-name=tax_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=a100

# ============================================================
# Taxonomic Level Analysis for TaxaAdapter - CVPR Rebuttal
# Demonstrates how different taxonomic levels encode visual traits
# ============================================================

echo "============================================"
echo "TaxaAdapter Taxonomic Level Analysis"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "============================================"

export PYTHONUNBUFFERED=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate apadapter

python inference_taxonomic_level_analysis.py \
    --ip_ckpt /home/karimimonsefi.1/BioAdapter/ip_adapter.bin \
    --json_file /home/karimimonsefi.1/BioAdapter/subset_inat_4_mapped.json \
    --out_dir /home/karimimonsefi.1/BioAdapter/taxonomic_analysis \
    --num_samples 10 \
    --model_type bioclip \
    --steps 50 \
    --guidance_scale 7.5 \
    --scale 1.0 \
    --seed 42

echo "============================================"
echo "Taxonomic Level Analysis Complete"
echo "============================================"
