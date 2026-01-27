"""
Cross-Attention Visualization for TaxaAdapter

This script visualizes the cross-attention maps between taxonomy tokens (from BioCLIP)
and spatial image features during generation. Inspired by ConceptAttention (ICML 2025)
and PartCraft (ECCV 2024).

Usage:
    python inference_attention_viz.py \
        --ip_ckpt ip_adapter.bin \
        --json_file subset_inat_200_mapped.json \
        --out_dir attention_viz_outputs \
        --num_samples 1 \
        --model_type bioclip
"""

import os
import argparse
from typing import List, Optional, Dict
from PIL import Image
import numpy as np
from collections import defaultdict
from functools import partial

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


# Global attention store
ATTENTION_STORE = None


class IPAttnProcessor2_0WithCapture(nn.Module):
    """
    IP-Adapter Attention Processor that captures cross-attention maps.
    This replaces the original processor to enable attention visualization.
    """
    def __init__(self, hidden_size, cross_attention_dim=None, scale=1.0, num_tokens=4, layer_name=""):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.scale = scale
        self.num_tokens = num_tokens
        self.layer_name = layer_name

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
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
            # Split text embeddings and IP-adapter embeddings
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

        # IP-Adapter processing
        if ip_hidden_states is not None:
            ip_key = self.to_k_ip(ip_hidden_states)
            ip_value = self.to_v_ip(ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            ip_hidden_states_out = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            # Capture attention map: compute explicitly
            # query: (batch, heads, seq_len, head_dim)
            # ip_key: (batch, heads, num_tokens, head_dim)
            with torch.no_grad():
                scale_factor = head_dim ** -0.5
                attn_weights = torch.matmul(query, ip_key.transpose(-2, -1)) * scale_factor
                attn_weights = F.softmax(attn_weights, dim=-1)
                # attn_weights: (batch, heads, seq_len, num_tokens)

                # Store attention if we have a store
                if ATTENTION_STORE is not None:
                    # Reshape for storage: (batch*heads, seq_len, num_tokens)
                    attn_to_store = attn_weights.permute(0, 1, 2, 3).reshape(
                        batch_size * attn.heads, -1, self.num_tokens
                    )
                    ATTENTION_STORE.store(self.layer_name, attn_to_store)

            ip_hidden_states_out = ip_hidden_states_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states_out = ip_hidden_states_out.to(query.dtype)

            hidden_states = hidden_states + self.scale * ip_hidden_states_out

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttentionStore:
    """Stores cross-attention maps during the diffusion process."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.attention_maps = defaultdict(list)
        self.current_timestep = 0

    def set_timestep(self, t):
        self.current_timestep = t

    def store(self, layer_name: str, attn_map: torch.Tensor):
        """Store attention map for a specific layer and timestep."""
        self.attention_maps[layer_name].append({
            'timestep': self.current_timestep,
            'attn': attn_map.detach().cpu()
        })

    def get_average_attention(self, resolution: int = 64, num_tokens: int = 4) -> torch.Tensor:
        """Aggregate attention maps across layers and timesteps."""
        aggregated = []

        print(f"  [DEBUG] Aggregating attention from {len(self.attention_maps)} layers")

        for layer_name, attn_list in self.attention_maps.items():
            for attn_dict in attn_list:
                attn = attn_dict['attn']  # (batch*heads, spatial, num_tokens)

                if attn.dim() != 3:
                    continue

                batch_heads, spatial, n_tokens = attn.shape

                if n_tokens != num_tokens:
                    continue

                # Determine number of heads (SD 1.5 uses 8 heads)
                num_heads = 8
                if batch_heads % num_heads != 0:
                    continue
                batch = batch_heads // num_heads

                # Reshape and average over heads
                attn = attn.view(batch, num_heads, spatial, n_tokens)
                attn = attn.mean(dim=1)  # (batch, spatial, num_tokens)

                # Get spatial resolution
                h = w = int(np.sqrt(spatial))
                if h * w != spatial:
                    continue

                # Reshape to spatial
                attn = attn.view(batch, h, w, n_tokens)
                attn = attn.permute(0, 3, 1, 2)  # (batch, num_tokens, h, w)

                # Resize to target resolution
                attn = F.interpolate(attn.float(), size=(resolution, resolution),
                                    mode='bilinear', align_corners=False)
                aggregated.append(attn)

        if not aggregated:
            print("  [DEBUG] No attention maps collected!")
            return None

        print(f"  [DEBUG] Aggregated {len(aggregated)} attention maps")
        aggregated = torch.cat(aggregated, dim=0)
        aggregated = aggregated.mean(dim=0)  # (num_tokens, H, W)

        return aggregated


def setup_attention_capture_processors(ip_model, num_tokens: int = 4):
    """
    Replace IP-Adapter processors with capturing versions.
    Returns the original processors for restoration.
    """
    unet = ip_model.pipe.unet
    original_processors = {}
    new_processors = {}

    for name, processor in unet.attn_processors.items():
        original_processors[name] = processor

        # Only replace cross-attention processors (not self-attention)
        if hasattr(processor, 'to_k_ip'):
            # This is an IP-Adapter processor
            new_proc = IPAttnProcessor2_0WithCapture(
                hidden_size=processor.hidden_size,
                cross_attention_dim=processor.cross_attention_dim,
                scale=processor.scale,
                num_tokens=num_tokens,
                layer_name=name,
            )
            # Copy weights
            new_proc.to_k_ip.load_state_dict(processor.to_k_ip.state_dict())
            new_proc.to_v_ip.load_state_dict(processor.to_v_ip.state_dict())
            new_proc = new_proc.to(ip_model.device, dtype=torch.float16)
            new_processors[name] = new_proc
        else:
            new_processors[name] = processor

    unet.set_attn_processor(new_processors)
    print(f"  [INFO] Replaced {sum(1 for p in new_processors.values() if isinstance(p, IPAttnProcessor2_0WithCapture))} IP-Adapter processors")

    return original_processors


def restore_processors(ip_model, original_processors):
    """Restore original attention processors."""
    ip_model.pipe.unet.set_attn_processor(original_processors)


# ---------- Visualization Functions ----------

def create_attention_heatmap(attention: torch.Tensor, image: Image.Image,
                            token_idx: int, alpha: float = 0.6) -> tuple:
    """Overlay attention heatmap on image."""
    attn = attention[token_idx].numpy()
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    img_size = image.size
    attn_resized = Image.fromarray((attn * 255).astype(np.uint8))
    attn_resized = attn_resized.resize(img_size, Image.BILINEAR)
    attn_resized = np.array(attn_resized) / 255.0

    cmap = cm.get_cmap('jet')
    heatmap = cmap(attn_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(heatmap)

    blended = Image.blend(image.convert('RGB'), heatmap_img, alpha)
    return blended, attn_resized


def visualize_all_tokens(image: Image.Image, attention: torch.Tensor,
                        taxonomy_name: str, save_path: str, num_tokens: int = 4):
    """Create a visualization grid showing attention for each token."""
    tax_levels = taxonomy_name.split()
    level_names = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    fig, axes = plt.subplots(2, num_tokens + 1, figsize=(4 * (num_tokens + 1), 8))

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Generated Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Taxonomy info
    tax_text = '\n'.join([f'{level_names[i]}: {tax_levels[i]}'
                         for i in range(min(len(tax_levels), len(level_names)))])
    axes[1, 0].text(0.1, 0.5, tax_text, fontsize=10, verticalalignment='center',
                   transform=axes[1, 0].transAxes, family='monospace')
    axes[1, 0].set_title('Taxonomy', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Attention maps for each token
    for i in range(num_tokens):
        blended, attn_raw = create_attention_heatmap(attention, image, i)
        axes[0, i + 1].imshow(blended)
        axes[0, i + 1].set_title(f'Token {i + 1}', fontsize=12, fontweight='bold')
        axes[0, i + 1].axis('off')

        im = axes[1, i + 1].imshow(attn_raw, cmap='jet')
        axes[1, i + 1].set_title(f'Attention Map {i + 1}', fontsize=10)
        axes[1, i + 1].axis('off')
        plt.colorbar(im, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    plt.suptitle(f'Cross-Attention Visualization\n{taxonomy_name}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_per_level_visualization(image: Image.Image, attention: torch.Tensor,
                                  taxonomy_name: str, save_path: str):
    """Create visualization mapping tokens to potential taxonomic meanings."""
    tax_levels = taxonomy_name.split()
    level_names = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    num_tokens = attention.shape[0]

    fig = plt.figure(figsize=(16, 10))

    # Main image
    ax_main = fig.add_axes([0.05, 0.35, 0.25, 0.55])
    ax_main.imshow(image)
    ax_main.set_title('Generated Image', fontsize=12, fontweight='bold')
    ax_main.axis('off')

    # Taxonomy hierarchy
    ax_tax = fig.add_axes([0.05, 0.05, 0.25, 0.25])
    hierarchy_text = "Taxonomic Hierarchy:\n" + "\n".join(
        [f"  {level_names[i]}: {tax_levels[i]}" for i in range(min(len(tax_levels), len(level_names)))]
    )
    ax_tax.text(0.05, 0.95, hierarchy_text, fontsize=9, verticalalignment='top',
               family='monospace', transform=ax_tax.transAxes)
    ax_tax.axis('off')

    # Attention maps in a grid
    for i in range(num_tokens):
        row = i // 2
        col = i % 2
        ax = fig.add_axes([0.35 + col * 0.32, 0.55 - row * 0.45, 0.28, 0.38])

        blended, _ = create_attention_heatmap(attention, image, i, alpha=0.5)
        ax.imshow(blended)

        potential_meaning = f"Token {i+1}\n(Higher-level)" if i < 2 else f"Token {i+1}\n(Fine-grained)"
        ax.set_title(potential_meaning, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f'Cross-Attention Analysis: {" ".join(tax_levels[-2:])}',
                fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------- Utilities ----------

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


def generate_with_attention_capture(ip_model, tokens, attention_store: AttentionStore,
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

    # Get embeddings
    image_prompt_embeds, uncond_image_prompt_embeds = ip_model.get_image_embeds(pil_image=tokens)

    with torch.inference_mode():
        prompt_embeds_, negative_prompt_embeds_ = ip_model.pipe.encode_prompt(
            [prompt],
            device=ip_model.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[negative_prompt],
        )
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

    generator = torch.Generator(device=ip_model.device).manual_seed(seed)

    ip_model.pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = ip_model.pipe.scheduler.timesteps

    latents = torch.randn(
        (1, ip_model.pipe.unet.config.in_channels, 64, 64),
        generator=generator,
        device=ip_model.device,
        dtype=torch.float16,
    )
    latents = latents * ip_model.pipe.scheduler.init_noise_sigma

    for i, t in enumerate(timesteps):
        attention_store.set_timestep(t.item())

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = ip_model.pipe.scheduler.scale_model_input(latent_model_input, t)

        with torch.inference_mode():
            noise_pred = ip_model.pipe.unet(
                latent_model_input,
                t,
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


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser("Cross-Attention Visualization for TaxaAdapter")
    p.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--ip_ckpt", required=True, help="Path to ip_adapter.bin")

    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="attention_viz_outputs")
    p.add_argument("--num_tokens", type=int, default=4)
    p.add_argument("--taxonomic_prompt", action="store_true")

    p.add_argument("--json_file", required=True)
    p.add_argument("--model_type", type=str, default="bioclip", choices=["bioclip", "taxabind", "clip"])
    p.add_argument("--resolution", type=int, default=64)

    return p.parse_args()


# ---------- Main ----------

@torch.inference_mode()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TaxaAdapter Cross-Attention Visualization")
    print("Inspired by ConceptAttention (ICML 2025) and PartCraft (ECCV 2024)")
    print("=" * 60)

    print("\n[1/5] Loading models...")
    pipe = make_pipe(args.base_model, args.vae_model, device)

    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    bioclip_tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

    config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
    taxabind = TaxaBind(config)
    location_encoder = taxabind.get_location_encoder().eval()
    taxabind_image_text_model = taxabind.get_image_text_encoder().eval()
    taxabind_tokenizer = taxabind.get_tokenizer()

    if args.model_type == "bioclip":
        tokenizer = bioclip_tok
    elif args.model_type == "taxabind":
        tokenizer = taxabind_tokenizer

    ip_model = IPAdapter(
        pipe,
        image_encoder_path=None,
        ip_ckpt=args.ip_ckpt,
        device=device,
        model_type=args.model_type,
        bioclip=bioclip_model,
        taxabind=taxabind_image_text_model,
        location_encoder=location_encoder,
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

    print(f"Found {len(unique_items)} unique taxonomic names")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[4/5] Generating visualizations for {min(args.num_samples, len(unique_items))} samples...")

    attention_store = AttentionStore()

    for idx, entry in tqdm(enumerate(unique_items[:args.num_samples]),
                          total=min(args.num_samples, len(unique_items))):
        taxa_name = entry["taxonomic_name"]

        if args.model_type in ["bioclip", "taxabind"]:
            tokens = tokenizer(taxa_name).to(device)

        if args.taxonomic_prompt:
            prompt = f"best quality, high quality photo of {taxa_name}"
        else:
            prompt = "best quality, high quality"

        # Generate with attention capture
        image = generate_with_attention_capture(
            ip_model, tokens, attention_store,
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            scale=args.scale,
            num_tokens=args.num_tokens,
        )

        # Get aggregated attention
        attention = attention_store.get_average_attention(
            resolution=args.resolution,
            num_tokens=args.num_tokens
        )

        if attention is None:
            print(f"Warning: No attention captured for {taxa_name}")
            continue

        # Create safe filename
        safe_name = taxa_name.replace(" ", "_").replace("/", "-")[:50]
        sample_dir = os.path.join(args.out_dir, safe_name)
        os.makedirs(sample_dir, exist_ok=True)

        # Save generated image
        image.save(os.path.join(sample_dir, "generated.png"))

        # Create visualizations
        visualize_all_tokens(
            image, attention, taxa_name,
            os.path.join(sample_dir, "attention_all_tokens.png"),
            num_tokens=args.num_tokens
        )

        create_per_level_visualization(
            image, attention, taxa_name,
            os.path.join(sample_dir, "attention_per_level.png")
        )

        # Individual token heatmaps
        for token_idx in range(args.num_tokens):
            blended, attn_raw = create_attention_heatmap(attention, image, token_idx)
            blended.save(os.path.join(sample_dir, f"attention_token_{token_idx+1}_overlay.png"))

            plt.figure(figsize=(6, 6))
            plt.imshow(attn_raw, cmap='jet')
            plt.colorbar()
            plt.title(f'Token {token_idx+1} Attention')
            plt.axis('off')
            plt.savefig(os.path.join(sample_dir, f"attention_token_{token_idx+1}_raw.png"),
                       dpi=150, bbox_inches='tight')
            plt.close()

        print(f"  Saved visualizations for: {taxa_name}")

    print("[5/5] Restoring original processors...")
    restore_processors(ip_model, original_processors)

    print("\n" + "=" * 60)
    print(f"Done! Visualizations saved to: {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
