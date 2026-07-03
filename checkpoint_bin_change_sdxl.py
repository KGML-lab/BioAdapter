import os, sys, torch
from safetensors.torch import load_file, save_file
from diffusers import UNet2DConditionModel

# --- EDIT THESE ---
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL base model
CKPT_DIR   = "/scratch/bio_diffusion/ip-adapter_runs/model_runs/supplementary/sdxl_BioCLIP_birds_subset_old_gpus2_1024_24hrs_batch_12/checkpoint-65000"  # EDIT THIS
IN_SFT     = os.path.join(CKPT_DIR, "model.safetensors")
OUT_BIN    = os.path.join(CKPT_DIR, "ip_adapter.bin")
OUT_SFT    = os.path.join(CKPT_DIR, "ip_adapter.safetensors")

# --- Load saved weights ---
sd = load_file(IN_SFT, device="cpu")

# projector params live under image_proj_model.*
image_proj_sd = {k.replace("image_proj_model.", "", 1): v
                 for k, v in sd.items() if k.startswith("image_proj_model.")}
print("projector param tensors:", len(image_proj_sd))  # expect 4 (proj/norm {weight,bias})

# Build a reference UNet to get the exact attn processor order (so indices match)
print(f"Loading reference UNet from {BASE_MODEL}...")
unet_ref = UNet2DConditionModel.from_pretrained(BASE_MODEL, subfolder="unet")
print(f"UNet cross_attention_dim: {unet_ref.config.cross_attention_dim}")  # Should be 2048 for SDXL
names_ordered = list(unet_ref.attn_processors.keys())
bases = [n[:-len(".processor")] if n.endswith(".processor") else n for n in names_ordered]

# IP weights are stored under: unet.<base>.processor.to_{k,v}_ip.weight
ip_adapter_sd = {}
missing_cnt = 0
for idx, base in enumerate(bases):
    k_key = f"unet.{base}.processor.to_k_ip.weight"
    v_key = f"unet.{base}.processor.to_v_ip.weight"
    if k_key in sd and v_key in sd:
        ip_adapter_sd[f"{idx}.to_k_ip.weight"] = sd[k_key]
        ip_adapter_sd[f"{idx}.to_v_ip.weight"] = sd[v_key]
    else:
        # many attn1 (self-attn) layers don't have IP params — that's fine
        missing_cnt += 1

print("ip-adapter param tensors:", len(ip_adapter_sd), "(layers without IP weights:", missing_cnt, ")")
if not image_proj_sd or not ip_adapter_sd:
    # help you debug prefixes quickly
    print("\nExample keys containing 'processor' or 'image_proj_model':")
    shown = 0
    for k in sd.keys():
        if "processor" in k or "image_proj_model" in k:
            print("  ", k)
            shown += 1
            if shown >= 30: break
    sys.exit("No params found; adjust prefixes or BASE_MODEL/CKPT_DIR.")

# --- Save nested .bin (your loader's non-safetensors path) ---
torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_adapter_sd}, OUT_BIN)
print("Wrote", OUT_BIN)

# --- Also save flat .safetensors (your loader's safetensors path) ---
flat = {f"image_proj.{k}": v for k, v in image_proj_sd.items()}
flat.update({f"ip_adapter.{k}": v for k, v in ip_adapter_sd.items()})
save_file(flat, OUT_SFT)
print("Wrote", OUT_SFT)
