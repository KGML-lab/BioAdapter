#!/bin/bash
#SBATCH --job-name=attn_viz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=a100

# ============================================================
# Cross-Attention Visualization for TaxaAdapter
# For CVPR Rebuttal - Reviewer Question on Attention Maps
# ============================================================

echo "============================================"
echo "TaxaAdapter Attention Visualization"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $(pwd)"
echo "============================================"

export PYTHONUNBUFFERED=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate apadapter

# Run attention visualization for a subset of species
python inference_attention_viz.py \
    --ip_ckpt /home/karimimonsefi.1/BioAdapter/ip_adapter.bin \
    --json_file /home/karimimonsefi.1/BioAdapter/subset_inat_4_mapped.json \
    --out_dir /home/karimimonsefi.1/BioAdapter/attention_outputs \
    --num_samples 20 \
    --model_type bioclip \
    --steps 50 \
    --guidance_scale 7.5 \
    --scale 1.0 \
    --taxonomic_prompt \
    --seed 42

echo "============================================"
echo "Attention Visualization Complete"
echo "============================================"
