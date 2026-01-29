"""
Cross-Attention Visualization for TaxaAdapter - CVPR Rebuttal Version

This script addresses the reviewer's question:
"Whether specific taxonomy tokens encoded specific visual traits"

Approach:
- Generate images using progressively more specific taxonomy (Kingdom → Species)
- Visualize cross-attention maps at each level
- Show that finer taxonomy focuses attention on more discriminative visual regions

Inspired by ConceptAttention (ICML 2025) and PartCraft (ECCV 2024).

Usage:
    python inference_attention_viz_final.py \
        --ip_ckpt ip_adapter.bin \
        --json_file subset_inat_200_mapped.json \
        --out_dir attention_rebuttal \
        --num_samples 5 \
        --model_type bioclip
"""

import os
import argparse
from typing import List, Optional, Dict
from PIL import Image
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL

import open_clip
from transformers import PretrainedConfig
from rshf.taxabind import TaxaBind

from ip_adapter import IPAdapter
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec


TAXONOMIC_LEVELS = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
ATTENTION_STORE = None


class IPAttnProcessor2_0WithCapture(nn.Module):
    """IP-Adapter Attention Processor with cross-attention capture."""
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, layer_name=""):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.layer_name = layer_name
        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                attention_mask=None, temb=None, *args, **kwargs):
        global ATTENTION_STORE
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        ip_hidden_states = None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ip_hidden_states_out = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            # Capture cross-attention map
            with torch.no_grad():
                scale_factor = head_dim ** -0.5
                attn_weights = torch.matmul(query, ip_key.transpose(-2, -1)) * scale_factor
                attn_weights = F.softmax(attn_weights, dim=-1)
                if ATTENTION_STORE is not None:
                    attn_to_store = attn_weights.reshape(batch_size * attn.heads, -1, self.num_tokens)
                    ATTENTION_STORE.store(self.layer_name, attn_to_store)

            ip_hidden_states_out = ip_hidden_states_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states_out = ip_hidden_states_out.to(query.dtype)
            hidden_states = hidden_states + self.scale * ip_hidden_states_out

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class AttentionStore:
    def __init__(self):
        self.reset()

    def reset(self):
        self.attention_maps = defaultdict(list)

    def store(self, layer_name: str, attn_map: torch.Tensor):
        self.attention_maps[layer_name].append(attn_map.detach().cpu())

    def get_averaged_attention(self, resolution: int = 64, num_tokens: int = 4) -> np.ndarray:
        """Get attention averaged over all tokens, layers, and timesteps."""
        aggregated = []
        for layer_name, attn_list in self.attention_maps.items():
            for attn in attn_list:
                if attn.dim() != 3:
                    continue
                batch_heads, spatial, n_tokens = attn.shape
                if n_tokens != num_tokens:
                    continue
                num_heads = 8
                if batch_heads % num_heads != 0:
                    continue
                batch = batch_heads // num_heads
                attn = attn.view(batch, num_heads, spatial, n_tokens)
                attn = attn.mean(dim=1)  # Average over heads
                attn = attn.mean(dim=-1)  # Average over tokens
                h = w = int(np.sqrt(spatial))
                if h * w != spatial:
                    continue
                attn = attn.view(batch, h, w)
                attn = F.interpolate(attn.unsqueeze(1).float(), size=(resolution, resolution),
                                    mode='bilinear', align_corners=False).squeeze(1)
                aggregated.append(attn)
        if not aggregated:
            return None
        aggregated = torch.cat(aggregated, dim=0)
        return aggregated.mean(dim=0).numpy()


def setup_processors(ip_model, num_tokens: int = 4):
    unet = ip_model.pipe.unet
    original_processors = {}
    new_processors = {}
    for name, processor in unet.attn_processors.items():
        original_processors[name] = processor
        if hasattr(processor, 'to_k_ip'):
            new_proc = IPAttnProcessor2_0WithCapture(
                hidden_size=processor.hidden_size,
                cross_attention_dim=processor.cross_attention_dim,
                scale=processor.scale,
                num_tokens=num_tokens,
                layer_name=name,
            )
            new_proc.to_k_ip.load_state_dict(processor.to_k_ip.state_dict())
            new_proc.to_v_ip.load_state_dict(processor.to_v_ip.state_dict())
            new_proc = new_proc.to(ip_model.device, dtype=torch.float16)
            new_processors[name] = new_proc
        else:
            new_processors[name] = processor
    unet.set_attn_processor(new_processors)
    return original_processors


def restore_processors(ip_model, original_processors):
    ip_model.pipe.unet.set_attn_processor(original_processors)


def make_scheduler():
    return DDIMScheduler(
        num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1,
    )


def make_pipe(base_model, vae_model, device):
    vae = AutoencoderKL.from_pretrained(vae_model).to(dtype=torch.float16) if vae_model else None
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16, scheduler=make_scheduler(),
        vae=vae, feature_extractor=None, safety_checker=None,
    ).to(device)
    return pipe


def generate_with_attention(ip_model, tokens, attention_store, prompt, steps=50,
                           guidance_scale=7.5, seed=42, scale=1.0, num_tokens=4):
    global ATTENTION_STORE
    ATTENTION_STORE = attention_store
    attention_store.reset()
    ip_model.set_scale(scale)

    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    image_prompt_embeds, uncond_image_prompt_embeds = ip_model.get_image_embeds(pil_image=tokens)

    with torch.inference_mode():
        prompt_embeds_, negative_prompt_embeds_ = ip_model.pipe.encode_prompt(
            [prompt], device=ip_model.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt=[negative_prompt],
        )
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

    generator = torch.Generator(device=ip_model.device).manual_seed(seed)
    ip_model.pipe.scheduler.set_timesteps(steps)
    timesteps = ip_model.pipe.scheduler.timesteps

    latents = torch.randn((1, ip_model.pipe.unet.config.in_channels, 64, 64),
                         generator=generator, device=ip_model.device, dtype=torch.float16)
    latents = latents * ip_model.pipe.scheduler.init_noise_sigma

    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = ip_model.pipe.scheduler.scale_model_input(latent_model_input, t)
        with torch.inference_mode():
            noise_pred = ip_model.pipe.unet(
                latent_model_input, t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
            ).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = ip_model.pipe.scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / ip_model.pipe.vae.config.scaling_factor * latents
    with torch.inference_mode():
        image = ip_model.pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype(np.uint8)
    image = Image.fromarray(image[0])

    ATTENTION_STORE = None
    return image


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def create_overlay(image, attention, alpha=0.5):
    """Create attention overlay on image."""
    attn_norm = normalize(attention)
    attn_resized = np.array(Image.fromarray((attn_norm * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR)) / 255.0
    cmap = cm.get_cmap('jet')
    heatmap = (cmap(attn_resized)[:, :, :3] * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap)
    return Image.blend(image.convert('RGB'), heatmap_img, alpha), attn_resized


def create_rebuttal_figure(images_by_level, attentions_by_level, full_taxonomy, save_path):
    """
    Create the MAIN figure for CVPR rebuttal.

    Shows: Generated images and attention maps at each taxonomic level,
    demonstrating that finer taxonomy focuses attention on discriminative regions.
    """
    tax_parts = full_taxonomy.split()
    num_levels = len(tax_parts)
    species_name = f"{tax_parts[-2]} {tax_parts[-1]}" if len(tax_parts) >= 2 else full_taxonomy

    fig = plt.figure(figsize=(3 * num_levels, 8))
    gs = GridSpec(3, num_levels, figure=fig, height_ratios=[1, 1, 0.3], hspace=0.15, wspace=0.1)

    # Row 1: Generated images
    for level in range(1, num_levels + 1):
        ax = fig.add_subplot(gs[0, level - 1])
        if level in images_by_level:
            ax.imshow(images_by_level[level])
        ax.set_title(f'{TAXONOMIC_LEVELS[level-1]}\n"{tax_parts[level-1]}"', fontsize=9)
        ax.axis('off')

    # Row 2: Attention overlays
    for level in range(1, num_levels + 1):
        ax = fig.add_subplot(gs[1, level - 1])
        if level in images_by_level and level in attentions_by_level and attentions_by_level[level] is not None:
            overlay, _ = create_overlay(images_by_level[level], attentions_by_level[level], alpha=0.6)
            ax.imshow(overlay)
        ax.axis('off')

    # Row 3: Attention statistics bar
    entropies = []
    stds = []
    for level in range(1, num_levels + 1):
        if level in attentions_by_level and attentions_by_level[level] is not None:
            attn = normalize(attentions_by_level[level])
            stds.append(np.std(attn))
            attn_flat = attn.flatten()
            attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
            entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
            entropies.append(entropy)
        else:
            stds.append(0)
            entropies.append(0)

    # Show focus metric (std) as bars
    ax_bar = fig.add_subplot(gs[2, :])
    x = np.arange(num_levels)
    bars = ax_bar.bar(x, stds, color='steelblue', alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([TAXONOMIC_LEVELS[i][:3] for i in range(num_levels)], fontsize=8)
    ax_bar.set_ylabel('Focus\n(Std Dev)', fontsize=8)
    ax_bar.set_title('Attention becomes more focused with finer taxonomy', fontsize=9, style='italic')

    plt.suptitle(f'Cross-Attention Analysis: {species_name}\n'
                f'Row 1: Generated images | Row 2: Attention overlays (averaged over 4 tokens)',
                fontsize=11, fontweight='bold')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def create_difference_figure(attentions_by_level, full_taxonomy, save_path):
    """
    Show DIFFERENCE maps - what each taxonomic level adds.
    This directly answers: "Do specific taxonomy tokens encode specific visual traits?"
    """
    tax_parts = full_taxonomy.split()
    num_levels = len(tax_parts)

    fig, axes = plt.subplots(1, num_levels, figsize=(3 * num_levels, 3.5))

    # First level: just show the attention
    if 1 in attentions_by_level and attentions_by_level[1] is not None:
        axes[0].imshow(normalize(attentions_by_level[1]), cmap='jet')
        axes[0].set_title(f'{TAXONOMIC_LEVELS[0]}\n(Base attention)', fontsize=9)
    axes[0].axis('off')

    # Subsequent levels: show difference from previous
    for level in range(2, num_levels + 1):
        if (level in attentions_by_level and attentions_by_level[level] is not None and
            level - 1 in attentions_by_level and attentions_by_level[level - 1] is not None):
            curr = normalize(attentions_by_level[level])
            prev = normalize(attentions_by_level[level - 1])
            diff = curr - prev
            im = axes[level - 1].imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            axes[level - 1].set_title(f'+{TAXONOMIC_LEVELS[level-1]}\n(Δ attention)', fontsize=9)
        axes[level - 1].axis('off')

    plt.suptitle(f'Attention Difference Maps: What each taxonomic level adds\n'
                f'Red = increased focus | Blue = decreased focus',
                fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def parse_args():
    p = argparse.ArgumentParser("Cross-Attention Visualization for CVPR Rebuttal")
    p.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--ip_ckpt", required=True)
    p.add_argument("--json_file", required=True)
    p.add_argument("--out_dir", default="attention_rebuttal")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--model_type", default="bioclip", choices=["bioclip", "taxabind"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--resolution", type=int, default=64)
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("TaxaAdapter Cross-Attention Visualization")
    print("For CVPR Rebuttal: Demonstrating taxonomy encodes visual traits")
    print("=" * 70)

    print("\n[1/4] Loading models...")
    pipe = make_pipe(args.base_model, args.vae_model, device)

    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    bioclip_tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

    config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
    taxabind = TaxaBind(config)
    location_encoder = taxabind.get_location_encoder().eval()
    taxabind_image_text_model = taxabind.get_image_text_encoder().eval()
    taxabind_tokenizer = taxabind.get_tokenizer()

    tokenizer = bioclip_tok if args.model_type == "bioclip" else taxabind_tokenizer

    ip_model = IPAdapter(
        pipe, image_encoder_path=None, ip_ckpt=args.ip_ckpt, device=device,
        model_type=args.model_type, bioclip=bioclip_model,
        taxabind=taxabind_image_text_model, location_encoder=location_encoder,
        num_tokens=args.num_tokens,
    )

    print("[2/4] Setting up attention capture...")
    original_processors = setup_processors(ip_model, args.num_tokens)
    attention_store = AttentionStore()

    print("[3/4] Loading data...")
    with open(args.json_file) as f:
        items = json.load(f)
    seen = set()
    unique_items = []
    for ex in items:
        tax = ex["taxonomic_name"]
        if tax not in seen:
            seen.add(tax)
            unique_items.append(ex)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[4/4] Generating visualizations for {min(args.num_samples, len(unique_items))} species...")

    for idx, entry in tqdm(enumerate(unique_items[:args.num_samples]),
                          total=min(args.num_samples, len(unique_items))):
        full_taxonomy = entry["taxonomic_name"]
        tax_parts = full_taxonomy.split()
        num_levels = min(len(tax_parts), 7)

        safe_name = full_taxonomy.replace(" ", "_").replace("/", "-")[:50]
        sample_dir = os.path.join(args.out_dir, safe_name)
        os.makedirs(sample_dir, exist_ok=True)

        images_by_level = {}
        attentions_by_level = {}

        # Generate for each taxonomic level
        for level in range(1, num_levels + 1):
            partial_taxonomy = " ".join(tax_parts[:level])
            tokens = tokenizer(partial_taxonomy).to(device)
            prompt = f"best quality, high quality photo of {partial_taxonomy}"

            image = generate_with_attention(
                ip_model, tokens, attention_store, prompt,
                steps=args.steps, guidance_scale=args.guidance_scale,
                seed=args.seed, scale=args.scale, num_tokens=args.num_tokens,
            )

            attention = attention_store.get_averaged_attention(
                resolution=args.resolution, num_tokens=args.num_tokens
            )

            images_by_level[level] = image
            attentions_by_level[level] = attention

            # Save individual images
            image.save(os.path.join(sample_dir, f"level_{level}_{TAXONOMIC_LEVELS[level-1]}.png"))

            # Save individual attention overlay
            if attention is not None:
                overlay, _ = create_overlay(image, attention, alpha=0.5)
                overlay.save(os.path.join(sample_dir, f"attention_{level}_{TAXONOMIC_LEVELS[level-1]}.png"))

        # Create main rebuttal figure
        create_rebuttal_figure(
            images_by_level, attentions_by_level, full_taxonomy,
            os.path.join(sample_dir, "rebuttal_figure.png")
        )

        # Create difference map figure
        create_difference_figure(
            attentions_by_level, full_taxonomy,
            os.path.join(sample_dir, "difference_maps.png")
        )

        print(f"  Saved: {full_taxonomy}")

    restore_processors(ip_model, original_processors)

    print("\n" + "=" * 70)
    print(f"Done! Figures saved to: {args.out_dir}")
    print("\nFor your rebuttal, use:")
    print("  - rebuttal_figure.png: Main visualization showing attention by level")
    print("  - difference_maps.png: Shows what each taxonomic level adds")
    print("=" * 70)


if __name__ == "__main__":
    main()
