#!/bin/bash
#SBATCH -J taxa_loc_seq_concat
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8                 # 4 procs × 8 workers
#SBATCH --time=23:30:00
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
# conda activate flux1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

echo "Starting accelerate..."


python inference.py --num_samples 10 --model_type bioclip \
--ip_ckpt /projects/ml4science/mridul/TaxaAadpter/model_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-380000/ip_adapter.bin \
--json /projects/ml4science/DATASETS/iNaturalist/subset_inat_200_mapped.json \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/inat_mini/taxadiffusion_inat200_compare_380k


python inference.py --num_samples 10 --model_type bioclip \
--ip_ckpt /projects/ml4science/mridul/TaxaAadpter/model_runs/tol-1m_4gpus_130hrs_1.4e4_64batch_512/checkpoint-370000/ip_adapter.bin \
--json /projects/ml4science/DATASETS/ToL-1m/new_tol1m_subset_1_500species_1instance.json \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/tol_1m/tol_1m_bioclip_subset1




python inference.py --num_samples 10 --model_type bioclip \
--ip_ckpt /projects/ml4science/mridul/TaxaAadpter/model_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-380000/ip_adapter.bin \
--json /home/mridul/code_bio_diffusion/evaluation-scripts/diversity_recall/group_jsons/inat_mini/insects_one_species_per_genus.json \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/per_genus/taxa/insects

python inference.py --num_samples 10 --model_type bioclip \
--ip_ckpt /projects/ml4science/mridul/TaxaAadpter/model_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-380000/ip_adapter.bin \
--json /home/mridul/code_bio_diffusion/evaluation-scripts/diversity_recall/group_jsons/inat_mini/aves_one_species_per_genus.json \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/per_genus/taxa/aves


# python inference.py --num_samples 10 --model_type bioclip \
# --ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-380000/ip_adapter.bin \
# --json /projects/ml4science/DATASETS/iNaturalist/train_mini_BIRDS_common_name_WITH_COORDS.json \
# --out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/inat_mini/taxadiffusion_inat200_compare_380k

# python inference.py --num_samples 10 --model_type bioclip \
# --ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-380000/ip_adapter.bin \
# --json /projects/ml4science/DATASETS/iNaturalist/train_mini_BIRDS_common_name_WITH_COORDS.json \
# --out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/inat_mini_birds_subset_levels_380k/level7

# python inference.py --num_samples 10 --model_type bioclip \
# --ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-380000/ip_adapter.bin \
# --json /projects/ml4science/DATASETS/iNaturalist/train_mini_BIRDS_common_name_WITH_COORDS.json \
# --out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/inat_mini_birds_subset_levels_380k/level7_TEST

# python inference.py --num_samples 10 --model_type bioclip \
# --ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-380000/ip_adapter.bin \
# --json /projects/ml4science/DATASETS/iNaturalist/subset_inat_200_mapped.json \
# --out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/inat_mini_birds_subset_levels_380k/subset_inat_200_mapped_seed_20 \
# --seed 20



python inference.py --num_samples 10 --model_type bioclip \
--ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/bioclip/tol-1m_4gpus_130hrs_1.4e4_64batch_512/checkpoint-370000/ip_adapter.bin \
--json /projects/ml4science/DATASETS/ToL-1m/longtail_subsets/subset_6to10_images.json \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/REBUTTAL/long_tailed/subset_6to10_images \
--dataset fishnet

python inference.py --num_samples 10 --model_type bioclip \
--ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/bioclip/tol-1m_4gpus_130hrs_1.4e4_64batch_512/checkpoint-370000/ip_adapter.bin \
--json /projects/ml4science/DATASETS/ToL-1m/new_tol1m_subset_1_500species_1instance.json \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/ECCV/bioclip-tol/tol-1m-subset1_10 \
--dataset fishnet

# ── bioclip_sae: same IP-Adapter checkpoint, but with SAE-reconstructed embeddings ──
# python inference.py --num_samples 10 --model_type bioclip_sae \
# --ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/bioclip/tol-1m_4gpus_130hrs_1.4e4_64batch_512/checkpoint-370000/ip_adapter.bin \
# --sae_ckpt /scratch/bio_diffusion/ip-adapter_runs/bioclip_sae/sae_model/sae.pt \
# --json /projects/ml4science/DATASETS/ToL-1m/longtail_subsets/subset_6to10_images.json \
# --out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/REBUTTAL/long_tailed/subset_6to10_images_SAE \
# --dataset fishnet


python inference.py --num_samples 10 --model_type taxabind \
--ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/model_runs/eccv/TaxaBind_inat-mini_8gpus_36hrs/checkpoint-80000/ip_adapter.bin \
--json /projects/ml4science/DATASETS/iNaturalist/subset_inat_200_mapped.json \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/ECCV/TaxaBind/TaxaBind_inat-mini_8gpus_36hrs_inat-200



python inference.py --num_samples 20 --model_type taxabind \
--ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/model_runs/eccv/TaxaBind_inat-mini_8gpus_36hrs/checkpoint-80000/ip_adapter.bin \
--json /projects/ml4science/DATASETS/iNaturalist/REBUTTAL/fine-grained_species.json \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/ECCV/ablations/fine-grained/taxa_bind_5 \
--guidance_scale 5.0

python inference.py --ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-380000/ip_adapter.bin \
--model_type bioclip --json /projects/ml4science/DATASETS/iNaturalist/REBUTTAL/fine-grained_species.json \
--num_samples 40 --out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/ECCV/ablations/fine-grained/bioclip_40 --guidance_scale 6 --steps 100


python inference.py --num_samples 10 --model_type taxabind \
--ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/model_runs/eccv/TaxaBind_tol-1m_4gpus_48hrs_epoch25/checkpoint-100000/ip_adapter.bin \
--json /projects/ml4science/DATASETS/ToL-1m/new_tol1m_subset_1_500species_1instance.json \
--out_dir /scratch/bio_diffusion/ip-adapter_runs/samples/ECCV/TaxaBind/TEST \
--dataset fishnet


# python inference_timestep_switch.py --model_type bioclip \  
# --ip_ckpt /scratch/bio_diffusion/ip-adapter_runs/inat_mini/bioclip_inat_mini_2gpus_126hrs/checkpoint-380000/ip_adapter.bin \
# --steps 50   --switch_steps 0 10 20 30 40 50 \
# --cond_a_text "Animalia Chordata Aves Charadriiformes Laridae Sterna hirundo" \
# --cond_b_text "Animalia Chordata Aves Charadriiformes Laridae Larus crassirostris" \
# --save_attention_maps   --viz_font_size 40   --viz_label_height 60 \
# --out_dir /scratch/bio_diffusion/new_experiments/timestep_sweeps/run3