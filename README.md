# TaxaAdapter

Abstract: 

---

## Installation

```bash
# Install dependencies
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/imageomics/rshf.git  # For TaxaBind

```

### Required Models

Download the following models before training/inference:
- [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [stabilityai/sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- BioCLIP-2: `hf-hub:imageomics/bioclip-2` (auto-downloaded)
- TaxaBind config: `MVRL/taxabind-config` (auto-downloaded)

---

## Training

### Basic Training Command

For training across 4 GPUs

```
accelerate launch --num_processes 4 --multi_gpu train.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --image_encoder_path="/path/to/image_encoder" \
  --data_json_file="/path/to/train.json" \
  --data_root_path="/path/to/images" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=64 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="/path/to/output" \
  --save_steps=2000 \
  --model_type="bioclip"
```

---

## Converting Checkpoints

After training, convert the checkpoint to adapter weights:

```bash
python checkpoint_bin_change.py --ckpt_dir /path/to/checkpoint-XXXXX
```

This creates:
- `taxa_adapter.bin` - PyTorch format
- `taxa_adapter.safetensors` - Safetensors format

Both formats are compatible with the inference script.

---


## Inference

### Basic Inference Command

```bash
python inference.py \
  --base_model="runwayml/stable-diffusion-v1-5" \
  --vae_model="stabilityai/sd-vae-ft-mse" \
  --ip_ckpt="/path/to/checkpoint/taxa_adapter.bin" \
  --json_file="/path/to/test.json" \
  --model_type="bioclip" \
  --dataset="inat" \
  --num_samples=10 \
  --steps=50 \
  --guidance_scale=3.5 \
  --seed=42 \
  --out_dir="outputs" \
  --num_tokens=4
```

