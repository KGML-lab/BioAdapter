"""
Taxonomic Level Analysis for TaxaAdapter

This script analyzes how different taxonomic levels (Kingdom → Species) encode
different visual traits by comparing cross-attention patterns when using
progressively more specific taxonomy information.

Inspired by ConceptAttention (ICML 2025) and PartCraft (ECCV 2024).

The analysis shows:
1. How attention patterns change as we add more specific taxonomic information
2. Which image regions become more/less attended with finer taxonomy
3. Difference maps showing what each taxonomic level adds

Usage:
    python inference_taxonomic_level_analysis.py \
        --ip_ckpt ip_adapter.bin \
        --json_file subset_inat_200_mapped.json \
        --out_dir taxonomic_analysis \
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

# Global attention store
ATTENTION_STORE = None


class IPAttnProcessor2_0WithCapture(nn.Module):
    """IP-Adapter Attention Processor that captures cross-attention maps."""
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

            # Capture attention map
            with torch.no_grad():
                scale_factor = head_dim ** -0.5
                attn_weights = torch.matmul(query, ip_key.transpose(-2, -1)) * scale_factor
                attn_weights = F.softmax(attn_weights, dim=-1)

                if ATTENTION_STORE is not None:
                    attn_to_store = attn_weights.reshape(
                        batch_size * attn.heads, -1, self.num_tokens
                    )
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
    """Stores cross-attention maps during diffusion."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.attention_maps = defaultdict(list)
        self.current_timestep = 0

    def set_timestep(self, t):
        self.current_timestep = t

    def store(self, layer_name: str, attn_map: torch.Tensor):
        self.attention_maps[layer_name].append({
            'timestep': self.current_timestep,
            'attn': attn_map.detach().cpu()
        })

    def get_average_attention(self, resolution: int = 64, num_tokens: int = 4) -> torch.Tensor:
        aggregated = []
        for layer_name, attn_list in self.attention_maps.items():
            for attn_dict in attn_list:
                attn = attn_dict['attn']
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
                attn = attn.mean(dim=1)
                h = w = int(np.sqrt(spatial))
                if h * w != spatial:
                    continue
                attn = attn.view(batch, h, w, n_tokens)
                attn = attn.permute(0, 3, 1, 2)
                attn = F.interpolate(attn.float(), size=(resolution, resolution),
                                    mode='bilinear', align_corners=False)
                aggregated.append(attn)
        if not aggregated:
            return None
        aggregated = torch.cat(aggregated, dim=0)
        aggregated = aggregated.mean(dim=0)
        return aggregated


def setup_attention_capture_processors(ip_model, num_tokens: int = 4):
    """Replace IP-Adapter processors with capturing versions."""
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


def make_scheduler() -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def make_pipe(base_model: str, vae_model: Optional[str], device: str) -> StableDiffusionPipeline:
    vae = AutoencoderKL.from_pretrained(vae_model).to(dtype=torch.float16) if vae_model else None
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        scheduler=make_scheduler(),
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    ).to(device)
    return pipe


def generate_with_attention(ip_model, tokens, attention_store: AttentionStore,
                           prompt: str, num_inference_steps: int = 50,
                           guidance_scale: float = 7.5, seed: int = 42,
                           scale: float = 1.0, num_tokens: int = 4):
    """Generate image while capturing cross-attention maps."""
    global ATTENTION_STORE
    ATTENTION_STORE = attention_store
    attention_store.reset()

    ip_model.set_scale(scale)

    if prompt is None:
        prompt = "best quality, high quality"
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
    ip_model.pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = ip_model.pipe.scheduler.timesteps

    latents = torch.randn(
        (1, ip_model.pipe.unet.config.in_channels, 64, 64),
        generator=generator, device=ip_model.device, dtype=torch.float16,
    )
    latents = latents * ip_model.pipe.scheduler.init_noise_sigma

    for i, t in enumerate(timesteps):
        attention_store.set_timestep(t.item())
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


def aggregate_token_attention(attention: torch.Tensor) -> np.ndarray:
    """Average attention across all tokens to get a single attention map."""
    if attention is None:
        return None
    return attention.mean(dim=0).numpy()


def normalize_attention(attn: np.ndarray) -> np.ndarray:
    """Normalize attention map to 0-1 range."""
    if attn is None:
        return None
    return (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)


def create_taxonomic_level_comparison(images_by_level: Dict[int, Image.Image],
                                     attentions_by_level: Dict[int, np.ndarray],
                                     full_taxonomy: str, save_path: str):
    """Create comprehensive visualization comparing attention at different taxonomic levels."""
    tax_parts = full_taxonomy.split()
    num_levels = min(len(tax_parts), 7)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, num_levels + 1, figure=fig, hspace=0.3, wspace=0.2)

    # Row 1: Generated images at each level
    for level in range(1, num_levels + 1):
        ax = fig.add_subplot(gs[0, level - 1])
        if level in images_by_level:
            ax.imshow(images_by_level[level])
        ax.set_title(f'{TAXONOMIC_LEVELS[level-1]}\n{tax_parts[level-1]}', fontsize=9)
        ax.axis('off')

    # Row 2: Attention maps at each level
    for level in range(1, num_levels + 1):
        ax = fig.add_subplot(gs[1, level - 1])
        if level in attentions_by_level and attentions_by_level[level] is not None:
            attn = normalize_attention(attentions_by_level[level])
            im = ax.imshow(attn, cmap='jet', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Attention\n(Levels 1-{level})', fontsize=9)
        ax.axis('off')

    # Row 3: Difference maps
    ax_info = fig.add_subplot(gs[2, 0])
    info_text = "Difference Maps:\nShows what each\nlevel adds to\nthe attention"
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10)
    ax_info.axis('off')

    for level in range(2, num_levels + 1):
        ax = fig.add_subplot(gs[2, level - 1])
        if (level in attentions_by_level and attentions_by_level[level] is not None and
            level - 1 in attentions_by_level and attentions_by_level[level - 1] is not None):
            attn_curr = normalize_attention(attentions_by_level[level])
            attn_prev = normalize_attention(attentions_by_level[level - 1])
            if attn_curr.shape != attn_prev.shape:
                attn_prev = np.array(Image.fromarray((attn_prev * 255).astype(np.uint8)).resize(
                    (attn_curr.shape[1], attn_curr.shape[0]), Image.BILINEAR)) / 255.0
            diff = attn_curr - attn_prev
            im = ax.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'+{TAXONOMIC_LEVELS[level-1]}', fontsize=9)
        ax.axis('off')

    # Full taxonomy at the end
    ax_full = fig.add_subplot(gs[0, num_levels])
    if num_levels in images_by_level:
        ax_full.imshow(images_by_level[num_levels])
    ax_full.set_title('Full Taxonomy', fontsize=10, fontweight='bold')
    ax_full.axis('off')

    ax_attn = fig.add_subplot(gs[1, num_levels])
    if num_levels in attentions_by_level and attentions_by_level[num_levels] is not None:
        attn = normalize_attention(attentions_by_level[num_levels])
        ax_attn.imshow(attn, cmap='jet')
    ax_attn.set_title('Full Attention', fontsize=10, fontweight='bold')
    ax_attn.axis('off')

    plt.suptitle(f'Taxonomic Level Analysis: {full_taxonomy}',
                fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_attention_evolution_plot(attentions_by_level: Dict[int, np.ndarray],
                                   full_taxonomy: str, save_path: str):
    """Create plot showing how attention focus evolves across taxonomic levels."""
    tax_parts = full_taxonomy.split()
    num_levels = min(len(tax_parts), 7)

    stats = {'mean': [], 'std': [], 'max': [], 'entropy': []}
    levels = []

    for level in range(1, num_levels + 1):
        if level in attentions_by_level and attentions_by_level[level] is not None:
            attn = normalize_attention(attentions_by_level[level])
            levels.append(level)
            stats['mean'].append(np.mean(attn))
            stats['std'].append(np.std(attn))
            stats['max'].append(np.max(attn))
            attn_flat = attn.flatten()
            attn_flat = attn_flat / attn_flat.sum()
            entropy = -np.sum(attn_flat * np.log(attn_flat + 1e-10))
            stats['entropy'].append(entropy)

    if not levels:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(levels, stats['mean'], 'b-o', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Taxonomic Level')
    axes[0, 0].set_ylabel('Mean Attention')
    axes[0, 0].set_title('Average Attention Intensity')
    axes[0, 0].set_xticks(levels)
    axes[0, 0].set_xticklabels([TAXONOMIC_LEVELS[l-1][:3] for l in levels], rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(levels, stats['std'], 'r-o', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Taxonomic Level')
    axes[0, 1].set_ylabel('Attention Std Dev')
    axes[0, 1].set_title('Attention Focus Sharpness')
    axes[0, 1].set_xticks(levels)
    axes[0, 1].set_xticklabels([TAXONOMIC_LEVELS[l-1][:3] for l in levels], rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(levels, stats['max'], 'g-o', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Taxonomic Level')
    axes[1, 0].set_ylabel('Max Attention')
    axes[1, 0].set_title('Peak Attention Intensity')
    axes[1, 0].set_xticks(levels)
    axes[1, 0].set_xticklabels([TAXONOMIC_LEVELS[l-1][:3] for l in levels], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(levels, stats['entropy'], 'm-o', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Taxonomic Level')
    axes[1, 1].set_ylabel('Attention Entropy')
    axes[1, 1].set_title('Attention Spread (Higher = More Diffuse)')
    axes[1, 1].set_xticks(levels)
    axes[1, 1].set_xticklabels([TAXONOMIC_LEVELS[l-1][:3] for l in levels], rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Attention Statistics Evolution\n{full_taxonomy}',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_heatmap_overlay_grid(image: Image.Image, attentions_by_level: Dict[int, np.ndarray],
                               full_taxonomy: str, save_path: str):
    """Create overlay visualization for each taxonomic level."""
    tax_parts = full_taxonomy.split()
    num_levels = min(len(tax_parts), 7)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    axes[0].imshow(image)
    axes[0].set_title('Generated Image', fontsize=10, fontweight='bold')
    axes[0].axis('off')

    for idx, level in enumerate(range(1, min(num_levels + 1, 8))):
        ax = axes[idx + 1] if idx + 1 < len(axes) else None
        if ax is None:
            break

        if level in attentions_by_level and attentions_by_level[level] is not None:
            attn = normalize_attention(attentions_by_level[level])
            attn_resized = np.array(Image.fromarray((attn * 255).astype(np.uint8)).resize(
                image.size, Image.BILINEAR)) / 255.0
            cmap = cm.get_cmap('jet')
            heatmap = cmap(attn_resized)[:, :, :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap)
            blended = Image.blend(image.convert('RGB'), heatmap_img, 0.5)
            ax.imshow(blended)
            ax.set_title(f'{TAXONOMIC_LEVELS[level-1]}\n({tax_parts[level-1]})', fontsize=9)
        ax.axis('off')

    plt.suptitle(f'Attention Overlay by Taxonomic Level\n{full_taxonomy}',
                fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def parse_args():
    p = argparse.ArgumentParser("Taxonomic Level Analysis for TaxaAdapter")
    p.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--ip_ckpt", required=True)

    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="taxonomic_analysis")
    p.add_argument("--num_tokens", type=int, default=4)

    p.add_argument("--json_file", required=True)
    p.add_argument("--model_type", type=str, default="bioclip", choices=["bioclip", "taxabind"])
    p.add_argument("--resolution", type=int, default=64)

    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("TaxaAdapter - Taxonomic Level Analysis")
    print("Analyzing how different taxonomic levels encode visual traits")
    print("=" * 70)

    print("\n[1/5] Loading models...")
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

    print("[2/5] Setting up attention capture processors...")
    original_processors = setup_attention_capture_processors(ip_model, num_tokens=args.num_tokens)

    print("[3/5] Loading data...")
    with open(args.json_file, "r") as f:
        items = json.load(f)

    seen = set()
    unique_items = []
    for ex in items:
        tax = ex["taxonomic_name"]
        if tax not in seen:
            seen.add(tax)
            unique_items.append(ex)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[4/5] Running taxonomic level analysis for {min(args.num_samples, len(unique_items))} species...")

    attention_store = AttentionStore()

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
                ip_model, tokens, attention_store,
                prompt=prompt, num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale, seed=args.seed, scale=args.scale,
                num_tokens=args.num_tokens,
            )

            attention = attention_store.get_average_attention(
                resolution=args.resolution, num_tokens=args.num_tokens
            )
            agg_attention = aggregate_token_attention(attention)

            images_by_level[level] = image
            attentions_by_level[level] = agg_attention

            image.save(os.path.join(sample_dir, f"level_{level}_{TAXONOMIC_LEVELS[level-1]}.png"))

        # Create visualizations
        create_taxonomic_level_comparison(
            images_by_level, attentions_by_level, full_taxonomy,
            os.path.join(sample_dir, "level_comparison.png")
        )

        create_attention_evolution_plot(
            attentions_by_level, full_taxonomy,
            os.path.join(sample_dir, "attention_statistics.png")
        )

        create_heatmap_overlay_grid(
            images_by_level[num_levels], attentions_by_level, full_taxonomy,
            os.path.join(sample_dir, "attention_overlays.png")
        )

        print(f"  Saved analysis for: {full_taxonomy}")

    print("[5/5] Restoring original processors...")
    restore_processors(ip_model, original_processors)

    print("\n" + "=" * 70)
    print(f"Analysis complete! Results saved to: {args.out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
