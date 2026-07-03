import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import open_clip
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from rshf.taxabind import TaxaBind
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig

from ip_adapter import IPAdapter
from ip_adapter.utils import (
    clear_attn_maps,
    clear_step_ip_contrib_maps,
    get_generator,
    get_net_ip_contrib_maps_per_sample,
    get_step_ip_contrib_maps_per_sample,
    register_cross_attention_hook,
)
from sae_utils import BioCLIPWithSAE, load_sae


DEFAULT_PROMPT = "best quality, high quality"
DEFAULT_NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"


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


def image_grid(imgs: List[Image.Image], rows: int, cols: int) -> Image.Image:
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, im in enumerate(imgs):
        grid.paste(im, box=((i % cols) * w, (i // cols) * h))
    return grid


def save_images(images: List[Image.Image], out_dir: str, grid_name: str = "grid.png", cols: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    for i, im in enumerate(images):
        im.save(os.path.join(out_dir, f"img_{i:02d}.png"))

    cols = min(cols, len(images))
    rows = (len(images) + cols - 1) // cols
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, im in enumerate(images):
        grid.paste(im, box=((i % cols) * w, (i // cols) * h))
    grid.save(os.path.join(out_dir, grid_name))


def slugify(value: Optional[str], fallback: str) -> str:
    if value is None:
        return fallback
    cleaned = value.strip().replace("/", "-")
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned[:120] or fallback


def clone_stage_bundle(stage_bundle: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in stage_bundle.items()}


def load_label_font(font_size: int) -> ImageFont.ImageFont:
    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_text_band(label: str, width: int, band_height: int = 40, font_size: int = 28) -> Image.Image:
    band = Image.new("RGB", (width, band_height), color=(245, 245, 245))
    draw = ImageDraw.Draw(band)
    font = load_label_font(font_size)
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = right - left
        text_h = bottom - top
    else:
        text_w, text_h = draw.textsize(label, font=font)
    draw.text(((width - text_w) // 2, max((band_height - text_h) // 2, 0)), label, fill=(0, 0, 0), font=font)
    return band


def make_labeled_tile(image: Image.Image, label: str, band_height: int = 40, font_size: int = 28) -> Image.Image:
    tile = Image.new("RGB", (image.width, image.height + band_height), color="white")
    tile.paste(make_text_band(label, image.width, band_height=band_height, font_size=font_size), (0, 0))
    tile.paste(image, (0, band_height))
    return tile


def stack_image_and_attention(image: Image.Image, attention_map: Optional[Image.Image]) -> Image.Image:
    if attention_map is None:
        return image

    attention_map = attention_map.convert("RGB").resize(image.size)
    combined = Image.new("RGB", (image.width, image.height * 2), color="white")
    combined.paste(image, (0, 0))
    combined.paste(attention_map, (0, image.height))
    return combined


def contribution_map_to_overlay(image: Image.Image, contribution_map: torch.Tensor, alpha: float = 0.72) -> Image.Image:
    contribution_map = contribution_map.detach().cpu().to(dtype=torch.float32)
    contribution_map = torch.clamp(contribution_map, 0.0, 1.0)
    heat = contribution_map.numpy()

    low = np.percentile(heat, 10.0)
    high = np.percentile(heat, 99.5)
    if high <= low:
        low = float(heat.min())
        high = float(heat.max())
    denom = max(high - low, 1e-6)
    heat = np.clip((heat - low) / denom, 0.0, 1.0)
    heat = np.power(heat, 0.75)
    heat_uint8 = (heat * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    heat_img = Image.fromarray(heat_rgb, mode="RGB").resize(image.size)
    base_gray = ImageOps.grayscale(image).convert("RGB")
    return Image.blend(base_gray, heat_img, alpha=alpha)


def save_labeled_grid(
    images: List[Image.Image],
    labels: List[str],
    out_path: str,
    cols: int,
    band_height: int = 40,
    font_size: int = 28,
):
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if not images:
        raise ValueError("images must be non-empty")

    labeled_tiles = [
        make_labeled_tile(image, label, band_height=band_height, font_size=font_size)
        for image, label in zip(images, labels)
    ]
    cols = min(cols, len(labeled_tiles))
    rows = (len(labeled_tiles) + cols - 1) // cols
    tile_w, tile_h = labeled_tiles[0].size
    grid = Image.new("RGB", size=(cols * tile_w, rows * tile_h), color="white")
    for idx, tile in enumerate(labeled_tiles):
        grid.paste(tile, box=((idx % cols) * tile_w, (idx // cols) * tile_h))
    grid.save(out_path)


def format_switch_label(step: int, total_steps: int) -> str:
    if step == total_steps:
        return "A"
    if step == 0:
        return "B"
    return f"switch@{step:02d}"


def format_switch_dir_name(step: int, total_steps: int) -> str:
    if step == total_steps:
        return "A_only"
    if step == 0:
        return "B_only"
    return f"switch_at_{step:02d}"


def format_timestep_label(step: int, switch_step: int, total_steps: int) -> str:
    if switch_step == 0:
        stage = "B"
    elif switch_step == total_steps:
        stage = "A"
    else:
        stage = "A" if step <= switch_step else "B"
    return f"t{step:02d}{stage}"


def format_short_switch_label(step: int, total_steps: int) -> str:
    if step == total_steps:
        return "A"
    if step == 0:
        return "B"
    return f"{step:02d}"


def format_difference_label(previous_step: int, current_step: int, total_steps: int) -> str:
    return f"{format_short_switch_label(previous_step, total_steps)}->{format_short_switch_label(current_step, total_steps)}"


def resolve_output_size(pipe: StableDiffusionPipeline, height: Optional[int], width: Optional[int]) -> Tuple[int, int]:
    default_size = pipe.unet.config.sample_size * pipe.vae_scale_factor
    return height or default_size, width or default_size


def should_record_timestep(step: int, total_steps: int, stride: int) -> bool:
    return step == total_steps or step % stride == 0


@torch.inference_mode()
def decode_latents_to_pil(pipe: StableDiffusionPipeline, latents: torch.Tensor) -> List[Image.Image]:
    latents = latents.detach().to(dtype=pipe.vae.dtype)
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * image.shape[0]
    return pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)


def difference_image_from_previous(
    previous_image: Image.Image,
    current_image: Image.Image,
    threshold: float = 0.0,
    overlay_alpha: float = 0.62,
) -> Image.Image:
    previous_rgb = np.asarray(previous_image.convert("RGB"), dtype=np.uint8)
    current_rgb = np.asarray(current_image.convert("RGB"), dtype=np.uint8)

    # Slight blur suppresses one-pixel noise before we compute perceptual differences.
    previous_rgb = cv2.GaussianBlur(previous_rgb, (5, 5), sigmaX=0.8)
    current_rgb = cv2.GaussianBlur(current_rgb, (5, 5), sigmaX=0.8)

    previous_lab = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    current_lab = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.linalg.norm(current_lab - previous_lab, axis=-1)

    low = np.percentile(diff, 25.0)
    high = np.percentile(diff, 99.5)
    if high <= low:
        if high <= 0:
            return current_image.convert("RGB")
        low = 0.0

    diff = np.clip((diff - low) / max(high - low, 1e-6), 0.0, 1.0)
    diff = np.power(diff, 0.7)
    diff = np.clip((diff - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    if diff.max() <= 0:
        return current_image.convert("RGB")

    # Turn the thresholded map into a soft mask so the overlay reads like
    # "changed regions" instead of a psychedelic full-frame color wash.
    mask = cv2.GaussianBlur(diff, (0, 0), sigmaX=1.2)
    mask = np.clip(mask, 0.0, 1.0)

    # Use only the hot end of the colormap so surviving changes read as
    # yellow/orange/red rather than blue/green.
    heat_uint8 = (160.0 + 95.0 * diff).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    base_rgb = np.asarray(current_image.convert("RGB"), dtype=np.float32)
    alpha_mask = (overlay_alpha * mask)[..., None]
    blended = base_rgb * (1.0 - alpha_mask) + heat_rgb.astype(np.float32) * alpha_mask
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8), mode="RGB")


def save_sweep_visualizations(
    results_by_step: Dict[int, Dict[str, object]],
    out_dir: str,
    total_steps: int,
    label_font_size: int = 28,
    label_band_height: int = 40,
):
    # Higher switch steps keep conditioning A for longer, so show them first.
    step_values = sorted(results_by_step, reverse=True)
    num_samples = len(results_by_step[step_values[0]]["images"])
    os.makedirs(out_dir, exist_ok=True)

    for sample_idx in range(num_samples):
        sample_images = []
        for step in step_values:
            image = results_by_step[step]["images"][sample_idx]
            contribution_maps = results_by_step[step]["contribution_maps"]
            contribution_map = None if contribution_maps is None else contribution_maps[sample_idx]
            overlay = None if contribution_map is None else contribution_map_to_overlay(image, contribution_map)
            sample_images.append(stack_image_and_attention(image, overlay))
        sample_labels = [format_switch_label(step, total_steps) for step in step_values]
        save_labeled_grid(
            sample_images,
            sample_labels,
            os.path.join(out_dir, f"sample_{sample_idx:02d}_across_switch_steps.png"),
            cols=len(sample_images),
            band_height=label_band_height,
            font_size=label_font_size,
        )


def save_switch_difference_visualizations(
    results_by_step: Dict[int, Dict[str, object]],
    out_dir: str,
    total_steps: int,
    threshold: float = 0.0,
    label_font_size: int = 28,
    label_band_height: int = 40,
):
    step_values = sorted(results_by_step)
    if len(step_values) < 2:
        return

    num_samples = len(results_by_step[step_values[0]]["images"])
    os.makedirs(out_dir, exist_ok=True)

    for sample_idx in range(num_samples):
        difference_images = []
        difference_labels = []

        for previous_step, current_step in zip(step_values[:-1], step_values[1:]):
            previous_image = results_by_step[previous_step]["images"][sample_idx]
            current_image = results_by_step[current_step]["images"][sample_idx]
            difference_images.append(
                difference_image_from_previous(
                    previous_image,
                    current_image,
                    threshold=threshold,
                )
            )
            difference_labels.append(format_difference_label(previous_step, current_step, total_steps))

        save_labeled_grid(
            difference_images,
            difference_labels,
            os.path.join(out_dir, f"sample_{sample_idx:02d}_difference_overlays.png"),
            cols=len(difference_images),
            band_height=label_band_height,
            font_size=label_font_size,
        )


def save_timestep_contribution_visualizations(
    images: List[Image.Image],
    timestep_contribution_maps: Optional[List[torch.Tensor]],
    saved_steps: List[int],
    timestep_images: Optional[List[List[Image.Image]]],
    out_dir: str,
    switch_step: int,
    total_steps: int,
    label_font_size: int = 28,
    label_band_height: int = 40,
):
    if not timestep_contribution_maps:
        return

    os.makedirs(out_dir, exist_ok=True)
    step_labels = [
        format_timestep_label(step_idx, switch_step, total_steps)
        for step_idx in saved_steps
    ]

    for sample_idx in range(len(images)):
        overlays = []
        for idx, step_map in enumerate(timestep_contribution_maps):
            base_image = images[sample_idx] if timestep_images is None else timestep_images[idx][sample_idx]
            overlays.append(contribution_map_to_overlay(base_image, step_map[sample_idx]))
        save_labeled_grid(
            overlays,
            step_labels,
            os.path.join(out_dir, f"sample_{sample_idx:02d}_contribution_overlays.png"),
            cols=min(10, len(overlays)),
            band_height=label_band_height,
            font_size=label_font_size,
        )


def save_timestep_image_visualizations(
    timestep_images: Optional[List[List[Image.Image]]],
    saved_steps: List[int],
    out_dir: str,
    switch_step: int,
    total_steps: int,
    label_font_size: int = 28,
    label_band_height: int = 40,
):
    if not timestep_images:
        return

    os.makedirs(out_dir, exist_ok=True)
    step_labels = [format_timestep_label(step_idx, switch_step, total_steps) for step_idx in saved_steps]

    for sample_idx in range(len(timestep_images[0])):
        images = [step_images[sample_idx] for step_images in timestep_images]
        save_labeled_grid(
            images,
            step_labels,
            os.path.join(out_dir, f"sample_{sample_idx:02d}_intermediate_images.png"),
            cols=min(10, len(images)),
            band_height=label_band_height,
            font_size=label_font_size,
        )


def parse_args():
    p = argparse.ArgumentParser(
        "Two-stage IP-Adapter inference: use conditioning A for the first N diffusion steps and conditioning B after."
    )
    p.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--ip_ckpt", required=True, help="Path to ip_adapter.bin or ip_adapter.safetensors")

    p.add_argument("--model_type", type=str, default="bioclip", choices=[
        "bioclip",
        "taxabind",
        "location",
        "clip",
        "taxa_loc_seq_concat",
        "loc_taxa_seq_concat",
        "bioclip_clip",
        "bioclip_sae",
        "biotrove",
    ])
    p.add_argument("--num_tokens", type=int, default=4, help="Must match training.")
    p.add_argument("--sae_ckpt", type=str, default="/scratch/bio_diffusion/ip-adapter_runs/bioclip_sae/sae_model/sae.pt")
    p.add_argument("--sae_alpha", type=float, default=0.5)

    p.add_argument("--steps", type=int, default=50)
    p.add_argument(
        "--first_stage_steps",
        type=int,
        default=20,
        help="Number of early denoising steps that use conditioning A. The remaining steps use conditioning B.",
    )
    p.add_argument(
        "--switch_steps",
        type=int,
        nargs="+",
        default=None,
        help="Optional sweep of switch steps, e.g. --switch_steps 5 10 15 20 25 30 35 40 45.",
    )
    p.add_argument("--guidance_scale", type=float, default=6)
    p.add_argument("--scale_a", type=float, default=1.0, help="IP-Adapter scale for the first stage.")
    p.add_argument("--scale_b", type=float, default=None, help="IP-Adapter scale for the second stage. Defaults to scale_a.")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)

    p.add_argument("--cond_a_text", type=str, default=None, help="Text condition for stage A.")
    p.add_argument("--cond_b_text", type=str, default=None, help="Text condition for stage B.")
    p.add_argument("--cond_a_latlon", type=float, nargs=2, metavar=("LAT", "LON"), default=None)
    p.add_argument("--cond_b_latlon", type=float, nargs=2, metavar=("LAT", "LON"), default=None)

    p.add_argument("--prompt_a", type=str, default=None, help="Text prompt used while stage A conditioning is active.")
    p.add_argument("--prompt_b", type=str, default=None, help="Text prompt used while stage B conditioning is active.")
    p.add_argument("--negative_prompt_a", type=str, default=None)
    p.add_argument("--negative_prompt_b", type=str, default=None)
    p.add_argument(
        "--save_attention_maps",
        action="store_true",
        help="If set, stack a per-image IP/BioCLIP contribution heatmap below each generated image in sweep visualizations.",
    )
    p.add_argument(
        "--save_timestep_attention_maps",
        action="store_true",
        help="If set, save a separate folder with per-sample IP/BioCLIP contribution maps across denoising timesteps.",
    )
    p.add_argument(
        "--save_timestep_images",
        action="store_true",
        help="If set, save a separate folder with decoded intermediate images across denoising timesteps.",
    )
    p.add_argument(
        "--save_switch_differences",
        action="store_true",
        help="If set in sweep mode, save per-sample image differences between consecutive switch settings.",
    )
    p.add_argument(
        "--switch_difference_threshold",
        type=float,
        default=0.0,
        help="Hide weak switch-difference overlays below this normalized threshold in [0, 1].",
    )
    p.add_argument(
        "--timestep_stride",
        type=int,
        default=5,
        help="Save every Nth denoising step for timestep visualizations, plus the final step.",
    )
    p.add_argument("--viz_font_size", type=int, default=28, help="Font size for sweep visualization labels.")
    p.add_argument("--viz_label_height", type=int, default=40, help="Height of the label band above each image.")

    p.add_argument("--out_dir", type=str, default="outputs_two_stage")
    p.add_argument("--save_name", type=str, default=None, help="Optional folder name inside out_dir.")
    return p.parse_args()


def validate_args(args):
    if args.scale_b is None:
        args.scale_b = args.scale_a
    if args.prompt_b is None:
        args.prompt_b = args.prompt_a
    if args.negative_prompt_b is None:
        args.negative_prompt_b = args.negative_prompt_a

    switch_steps = args.switch_steps if args.switch_steps is not None else [args.first_stage_steps]
    normalized_switch_steps = []
    for step in switch_steps:
        if step < 0 or step > args.steps:
            raise ValueError(f"Switch step must be between 0 and --steps ({args.steps}), got {step}.")
        normalized_switch_steps.append(step)
    args.switch_steps = sorted(set(normalized_switch_steps))
    args.first_stage_steps = args.switch_steps[0]
    if args.viz_font_size <= 0:
        raise ValueError("--viz_font_size must be > 0.")
    if args.viz_label_height <= 0:
        raise ValueError("--viz_label_height must be > 0.")
    if args.timestep_stride <= 0:
        raise ValueError("--timestep_stride must be > 0.")
    if not 0.0 <= args.switch_difference_threshold < 1.0:
        raise ValueError("--switch_difference_threshold must be in [0, 1).")

    text_models = {"bioclip", "taxabind", "clip", "taxa_loc_seq_concat", "loc_taxa_seq_concat", "bioclip_clip", "bioclip_sae", "biotrove"}
    location_models = {"location", "taxa_loc_seq_concat", "loc_taxa_seq_concat"}

    if args.model_type in text_models:
        if args.cond_a_text is None or args.cond_b_text is None:
            raise ValueError(f"--model_type {args.model_type} requires both --cond_a_text and --cond_b_text.")

    if args.model_type in location_models:
        if args.cond_a_latlon is None or args.cond_b_latlon is None:
            raise ValueError(f"--model_type {args.model_type} requires both --cond_a_latlon and --cond_b_latlon.")


def load_condition_models(args, device: str):
    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    bioclip_tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

    effective_model_type = args.model_type
    if effective_model_type == "bioclip_sae":
        print(f"Loading SAE from {args.sae_ckpt} ...")
        sae = load_sae(args.sae_ckpt, device="cpu")
        print(f"SAE blending alpha={args.sae_alpha} (0=pure SAE, 1=pure BioCLIP)")
        bioclip_model = BioCLIPWithSAE(bioclip_model, sae, alpha=args.sae_alpha)
        effective_model_type = "bioclip"

    config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
    taxabind = TaxaBind(config)
    location_encoder = taxabind.get_location_encoder().eval()
    taxabind_image_text_model = taxabind.get_image_text_encoder().eval()
    taxabind_tokenizer = taxabind.get_tokenizer()

    tokenizer = None
    secondary_tokenizer = None
    clip_text_with_proj = None

    if effective_model_type == "bioclip":
        tokenizer = bioclip_tokenizer
    elif effective_model_type in {"taxabind", "taxa_loc_seq_concat", "loc_taxa_seq_concat"}:
        tokenizer = taxabind_tokenizer
    elif effective_model_type == "clip":
        clip_ckpt = "openai/clip-vit-large-patch14"
        tokenizer = CLIPTokenizer.from_pretrained(clip_ckpt)
        clip_text_with_proj = CLIPTextModelWithProjection.from_pretrained(clip_ckpt).eval()
    elif effective_model_type == "bioclip_clip":
        clip_ckpt = "openai/clip-vit-large-patch14"
        tokenizer = CLIPTokenizer.from_pretrained(clip_ckpt)
        clip_text_with_proj = CLIPTextModelWithProjection.from_pretrained(clip_ckpt).eval()
        secondary_tokenizer = bioclip_tokenizer
    elif effective_model_type == "biotrove":
        bioclip_model = open_clip.create_model(
            "hf-hub:BGLab/BioTrove-CLIP",
            output_dict=True,
            require_pretrained=True,
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-16")

    bundle = {
        "bioclip_model": bioclip_model,
        "taxabind_image_text_model": taxabind_image_text_model,
        "location_encoder": location_encoder,
        "tokenizer": tokenizer,
        "secondary_tokenizer": secondary_tokenizer,
        "clip_text_with_proj": clip_text_with_proj,
        "effective_model_type": effective_model_type,
    }
    return bundle


def build_ip_model(args, pipe: StableDiffusionPipeline, device: str, bundle: Dict):
    model_type = bundle["effective_model_type"]

    if model_type == "clip":
        ip_model = IPAdapter(
            pipe,
            image_encoder_path=None,
            ip_ckpt=args.ip_ckpt,
            device=device,
            num_tokens=args.num_tokens,
            model_type=model_type,
            bioclip=bundle["clip_text_with_proj"],
            taxabind=bundle["taxabind_image_text_model"],
            location_encoder=bundle["location_encoder"],
        )
    elif model_type == "bioclip_clip":
        ip_model = IPAdapter(
            pipe,
            image_encoder_path=None,
            ip_ckpt=args.ip_ckpt,
            device=device,
            num_tokens=args.num_tokens,
            model_type=model_type,
            bioclip=bundle["bioclip_model"],
            taxabind=bundle["clip_text_with_proj"],
            location_encoder=bundle["location_encoder"],
        )
    else:
        ip_model = IPAdapter(
            pipe,
            image_encoder_path=None,
            ip_ckpt=args.ip_ckpt,
            device=device,
            num_tokens=args.num_tokens,
            model_type=model_type,
            bioclip=bundle["bioclip_model"],
            taxabind=bundle["taxabind_image_text_model"],
            location_encoder=bundle["location_encoder"],
        )
    return ip_model


def prepare_condition_tokens(
    model_type: str,
    text: Optional[str],
    latlon: Optional[Tuple[float, float]],
    tokenizer,
    secondary_tokenizer,
    device: str,
):
    location = None
    if latlon is not None:
        location = torch.tensor(latlon, dtype=torch.float32)

    if model_type in {"bioclip", "taxabind", "biotrove"}:
        return tokenizer(text).to(device)
    if model_type == "location":
        return location.unsqueeze(0).to(device)
    if model_type == "clip":
        tokens = tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return tokens.to(device)
    if model_type in {"taxa_loc_seq_concat", "loc_taxa_seq_concat"}:
        taxa_tokens = tokenizer(text).to(device)
        return taxa_tokens, location.unsqueeze(0).to(device)
    if model_type == "bioclip_clip":
        bioclip_tokens = secondary_tokenizer(text).to(device)
        clip_tokens = tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return bioclip_tokens, clip_tokens.to(device)

    raise ValueError(f"Unsupported model_type for two-stage inference: {model_type}")


@torch.inference_mode()
def build_stage_prompt_embeds(
    ip_model: IPAdapter,
    condition_tokens,
    prompt: Optional[str],
    negative_prompt: Optional[str],
    num_samples: int,
):
    prompt = prompt or DEFAULT_PROMPT
    negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT

    image_prompt_embeds, uncond_image_prompt_embeds = ip_model.get_image_embeds(pil_image=condition_tokens)
    bs_embed, seq_len, _ = image_prompt_embeds.shape

    image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
    image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

    uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
    uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

    prompt_embeds_text, negative_prompt_embeds_text = ip_model.pipe.encode_prompt(
        [prompt],
        device=ip_model.device,
        num_images_per_prompt=num_samples,
        do_classifier_free_guidance=True,
        negative_prompt=[negative_prompt],
    )

    prompt_embeds = torch.cat([prompt_embeds_text, image_prompt_embeds], dim=1)
    negative_prompt_embeds = torch.cat([negative_prompt_embeds_text, uncond_image_prompt_embeds], dim=1)
    cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    return {
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "cfg_prompt_embeds": cfg_prompt_embeds,
    }


def make_stage_switch_callback(
    ip_model: IPAdapter,
    first_stage_steps: int,
    late_stage_bundle: Dict[str, torch.Tensor],
    late_stage_scale: float,
    total_steps: int,
    timestep_stride: int,
    timestep_image_size: Optional[Tuple[int, int]] = None,
    num_samples: Optional[int] = None,
    saved_timestep_steps: Optional[List[int]] = None,
    timestep_contribution_maps: Optional[List[torch.Tensor]] = None,
    timestep_images: Optional[List[List[Image.Image]]] = None,
):
    switched = False

    def callback(_pipe, step_index: int, _timestep, _callback_kwargs):
        nonlocal switched
        callback_updates = {}
        step_number = step_index + 1
        should_record = saved_timestep_steps is not None and should_record_timestep(
            step_number, total_steps, timestep_stride
        )

        if should_record:
            saved_timestep_steps.append(step_number)
            if timestep_contribution_maps is not None:
                timestep_contribution_maps.append(
                    get_step_ip_contrib_maps_per_sample(
                        image_size=timestep_image_size,
                        num_samples=num_samples,
                    ).clone()
                )
            if timestep_images is not None:
                timestep_images.append(decode_latents_to_pil(_pipe, _callback_kwargs["latents"]))

        if timestep_contribution_maps is not None:
            clear_step_ip_contrib_maps()

        if switched or first_stage_steps <= 0 or first_stage_steps >= total_steps or step_number != first_stage_steps:
            return callback_updates

        switched = True
        ip_model.set_scale(late_stage_scale)
        print(f"Switching conditioning after step {step_index + 1} -> stage B")
        callback_updates["prompt_embeds"] = late_stage_bundle["cfg_prompt_embeds"].clone()
        callback_updates["negative_prompt_embeds"] = late_stage_bundle["negative_prompt_embeds"].clone()
        return callback_updates

    return callback


def write_run_metadata(args, effective_model_type: str, out_dir: str, first_stage_steps: int):
    metadata = {
        "model_type": effective_model_type,
        "steps": args.steps,
        "first_stage_steps": first_stage_steps,
        "second_stage_steps": args.steps - first_stage_steps,
        "guidance_scale": args.guidance_scale,
        "scale_a": args.scale_a,
        "scale_b": args.scale_b,
        "cond_a_text": args.cond_a_text,
        "cond_b_text": args.cond_b_text,
        "cond_a_latlon": args.cond_a_latlon,
        "cond_b_latlon": args.cond_b_latlon,
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "negative_prompt_a": args.negative_prompt_a,
        "negative_prompt_b": args.negative_prompt_b,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "height": args.height,
        "width": args.width,
        "save_attention_maps": args.save_attention_maps,
        "save_timestep_attention_maps": args.save_timestep_attention_maps,
        "save_timestep_images": args.save_timestep_images,
        "save_switch_differences": args.save_switch_differences,
        "switch_difference_threshold": args.switch_difference_threshold,
        "timestep_stride": args.timestep_stride,
        "viz_font_size": args.viz_font_size,
        "viz_label_height": args.viz_label_height,
        "ip_ckpt": args.ip_ckpt,
        "seed_reused_for_comparison": True,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def write_sweep_metadata(args, effective_model_type: str, out_dir: str):
    metadata = {
        "model_type": effective_model_type,
        "steps": args.steps,
        "switch_steps": args.switch_steps,
        "guidance_scale": args.guidance_scale,
        "scale_a": args.scale_a,
        "scale_b": args.scale_b,
        "cond_a_text": args.cond_a_text,
        "cond_b_text": args.cond_b_text,
        "cond_a_latlon": args.cond_a_latlon,
        "cond_b_latlon": args.cond_b_latlon,
        "prompt_a": args.prompt_a,
        "prompt_b": args.prompt_b,
        "negative_prompt_a": args.negative_prompt_a,
        "negative_prompt_b": args.negative_prompt_b,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "height": args.height,
        "width": args.width,
        "save_attention_maps": args.save_attention_maps,
        "save_timestep_attention_maps": args.save_timestep_attention_maps,
        "save_timestep_images": args.save_timestep_images,
        "save_switch_differences": args.save_switch_differences,
        "switch_difference_threshold": args.switch_difference_threshold,
        "timestep_stride": args.timestep_stride,
        "viz_font_size": args.viz_font_size,
        "viz_label_height": args.viz_label_height,
        "ip_ckpt": args.ip_ckpt,
        "seed_reused_for_comparison": True,
    }
    with open(os.path.join(out_dir, "sweep_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)


@torch.inference_mode()
def run_two_stage_generation(
    ip_model: IPAdapter,
    stage_a: Dict[str, torch.Tensor],
    stage_b: Dict[str, torch.Tensor],
    first_stage_steps: int,
    total_steps: int,
    guidance_scale: float,
    scale_a: float,
    scale_b: float,
    seed: int,
    height: Optional[int],
    width: Optional[int],
    capture_summary_attention_maps: bool = False,
    capture_timestep_attention_maps: bool = False,
    capture_timestep_images: bool = False,
    timestep_stride: int = 5,
):
    stage_a_local = clone_stage_bundle(stage_a)
    stage_b_local = clone_stage_bundle(stage_b)
    output_size = resolve_output_size(ip_model.pipe, height, width)
    saved_timestep_steps = [] if (capture_timestep_attention_maps or capture_timestep_images) else None
    timestep_contribution_maps = [] if capture_timestep_attention_maps else None
    timestep_images = [] if capture_timestep_images else None

    if first_stage_steps == 0:
        initial_stage = stage_b_local
        initial_scale = scale_b
    else:
        initial_stage = stage_a_local
        initial_scale = scale_a

    callback = None
    if capture_timestep_attention_maps or capture_timestep_images or (0 < first_stage_steps < total_steps):
        callback = make_stage_switch_callback(
            ip_model=ip_model,
            first_stage_steps=first_stage_steps,
            late_stage_bundle=stage_b_local,
            late_stage_scale=scale_b,
            total_steps=total_steps,
            timestep_stride=timestep_stride,
            timestep_image_size=output_size if capture_timestep_attention_maps else None,
            num_samples=stage_a_local["prompt_embeds"].shape[0] if capture_timestep_attention_maps else None,
            saved_timestep_steps=saved_timestep_steps,
            timestep_contribution_maps=timestep_contribution_maps,
            timestep_images=timestep_images,
        )

    ip_model.set_scale(initial_scale)
    generator = get_generator(seed, ip_model.device)

    print(f"Running {total_steps} steps: stage A for {first_stage_steps}, stage B for {total_steps - first_stage_steps}.")

    if capture_summary_attention_maps or capture_timestep_attention_maps:
        clear_attn_maps()

    images = ip_model.pipe(
        prompt_embeds=initial_stage["prompt_embeds"],
        negative_prompt_embeds=initial_stage["negative_prompt_embeds"],
        guidance_scale=guidance_scale,
        num_inference_steps=total_steps,
        generator=generator,
        height=height,
        width=width,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents", "prompt_embeds", "negative_prompt_embeds"],
    ).images

    contribution_maps = None
    if capture_summary_attention_maps:
        contribution_maps = get_net_ip_contrib_maps_per_sample(
            image_size=(images[0].height, images[0].width),
            num_samples=len(images),
        )

    return images, contribution_maps, saved_timestep_steps, timestep_contribution_maps, timestep_images


@torch.inference_mode()
def main():
    args = parse_args()
    validate_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = make_pipe(args.base_model, args.vae_model, device)
    bundle = load_condition_models(args, device)
    effective_model_type = bundle["effective_model_type"]
    ip_model = build_ip_model(args, pipe, device, bundle)
    capture_any_attention_maps = args.save_attention_maps or args.save_timestep_attention_maps
    if capture_any_attention_maps:
        ip_model.pipe.unet = register_cross_attention_hook(ip_model.pipe.unet)

    cond_a = prepare_condition_tokens(
        effective_model_type,
        args.cond_a_text,
        tuple(args.cond_a_latlon) if args.cond_a_latlon is not None else None,
        bundle["tokenizer"],
        bundle["secondary_tokenizer"],
        device,
    )
    cond_b = prepare_condition_tokens(
        effective_model_type,
        args.cond_b_text,
        tuple(args.cond_b_latlon) if args.cond_b_latlon is not None else None,
        bundle["tokenizer"],
        bundle["secondary_tokenizer"],
        device,
    )

    stage_a = build_stage_prompt_embeds(
        ip_model,
        cond_a,
        args.prompt_a,
        args.negative_prompt_a,
        args.num_samples,
    )
    stage_b = build_stage_prompt_embeds(
        ip_model,
        cond_b,
        args.prompt_b,
        args.negative_prompt_b,
        args.num_samples,
    )

    sweep_mode = len(args.switch_steps) > 1
    default_name = (
        f"{slugify(args.cond_a_text, 'stageA')}_to_{slugify(args.cond_b_text, 'stageB')}_"
        f"{args.switch_steps[0]}to{args.switch_steps[-1]}_sweep"
        if sweep_mode
        else f"{slugify(args.cond_a_text, 'stageA')}_to_{slugify(args.cond_b_text, 'stageB')}_{args.first_stage_steps}of{args.steps}"
    )
    save_name = args.save_name or default_name
    out_dir = os.path.join(args.out_dir, save_name)
    os.makedirs(out_dir, exist_ok=True)

    if sweep_mode:
        print(
            f"Sweeping switch steps {args.switch_steps} with seed {args.seed}. "
            "Each run reuses the same seed so sample indices are directly comparable."
        )
        write_sweep_metadata(args, effective_model_type, out_dir)
        results_by_step = {}

        for switch_step in args.switch_steps:
            images, contribution_maps, saved_timestep_steps, timestep_contribution_maps, timestep_images = run_two_stage_generation(
                ip_model=ip_model,
                stage_a=stage_a,
                stage_b=stage_b,
                first_stage_steps=switch_step,
                total_steps=args.steps,
                guidance_scale=args.guidance_scale,
                scale_a=args.scale_a,
                scale_b=args.scale_b,
                seed=args.seed,
                height=args.height,
                width=args.width,
                capture_summary_attention_maps=args.save_attention_maps,
                capture_timestep_attention_maps=args.save_timestep_attention_maps,
                capture_timestep_images=args.save_timestep_images,
                timestep_stride=args.timestep_stride,
            )
            results_by_step[switch_step] = {
                "images": images,
                "contribution_maps": contribution_maps,
            }
            if args.save_timestep_attention_maps:
                save_timestep_contribution_visualizations(
                    images=images,
                    timestep_contribution_maps=timestep_contribution_maps,
                    saved_steps=saved_timestep_steps,
                    timestep_images=timestep_images,
                    out_dir=os.path.join(out_dir, "timestep_contributions", format_switch_dir_name(switch_step, args.steps)),
                    switch_step=switch_step,
                    total_steps=args.steps,
                    label_font_size=args.viz_font_size,
                    label_band_height=args.viz_label_height,
                )
            if args.save_timestep_images:
                save_timestep_image_visualizations(
                    timestep_images=timestep_images,
                    saved_steps=saved_timestep_steps,
                    out_dir=os.path.join(out_dir, "timestep_images", format_switch_dir_name(switch_step, args.steps)),
                    switch_step=switch_step,
                    total_steps=args.steps,
                    label_font_size=args.viz_font_size,
                    label_band_height=args.viz_label_height,
                )
            print(f"Finished switch step {switch_step}")

        save_sweep_visualizations(
            results_by_step,
            out_dir,
            args.steps,
            label_font_size=args.viz_font_size,
            label_band_height=args.viz_label_height,
        )
        if args.save_switch_differences:
            save_switch_difference_visualizations(
                results_by_step,
                os.path.join(out_dir, "switch_differences"),
                args.steps,
                threshold=args.switch_difference_threshold,
                label_font_size=args.viz_font_size,
                label_band_height=args.viz_label_height,
            )
        print(f"Saved per-sample comparison strips -> {out_dir}")
    else:
        images, _, saved_timestep_steps, timestep_contribution_maps, timestep_images = run_two_stage_generation(
            ip_model=ip_model,
            stage_a=stage_a,
            stage_b=stage_b,
            first_stage_steps=args.first_stage_steps,
            total_steps=args.steps,
            guidance_scale=args.guidance_scale,
            scale_a=args.scale_a,
            scale_b=args.scale_b,
            seed=args.seed,
            height=args.height,
            width=args.width,
            capture_summary_attention_maps=False,
            capture_timestep_attention_maps=args.save_timestep_attention_maps,
            capture_timestep_images=args.save_timestep_images,
            timestep_stride=args.timestep_stride,
        )
        save_images(images, out_dir)
        write_run_metadata(args, effective_model_type, out_dir, args.first_stage_steps)
        if args.save_timestep_attention_maps:
            save_timestep_contribution_visualizations(
                images=images,
                timestep_contribution_maps=timestep_contribution_maps,
                saved_steps=saved_timestep_steps,
                timestep_images=timestep_images,
                out_dir=os.path.join(out_dir, "timestep_contributions"),
                switch_step=args.first_stage_steps,
                total_steps=args.steps,
                label_font_size=args.viz_font_size,
                label_band_height=args.viz_label_height,
            )
        if args.save_timestep_images:
            save_timestep_image_visualizations(
                timestep_images=timestep_images,
                saved_steps=saved_timestep_steps,
                out_dir=os.path.join(out_dir, "timestep_images"),
                switch_step=args.first_stage_steps,
                total_steps=args.steps,
                label_font_size=args.viz_font_size,
                label_band_height=args.viz_label_height,
            )
        print(f"Saved {len(images)} images to {out_dir}")


if __name__ == "__main__":
    main()
