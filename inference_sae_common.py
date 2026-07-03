import json
import os
from typing import Dict, Optional, Type

import open_clip
import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from rshf.taxabind import TaxaBind
from tqdm import tqdm
from transformers import PretrainedConfig

from ip_adapter import IPAdapter
from sae_utils import compute_embedding_debug_stats, load_sae, load_sae_from_dir


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


def save_images_per_class(images, save_dir, class_tag):
    os.makedirs(save_dir, exist_ok=True)
    for i, im in enumerate(images):
        im.save(os.path.join(save_dir, f"{class_tag}_sample{i:02d}.png"))


def save_images_fishnet(images, save_dir, taxonomic_name, num_samples):
    os.makedirs(save_dir, exist_ok=True)
    base_filename = taxonomic_name.replace(" ", "_")
    for i, im in enumerate(images):
        if num_samples == 1:
            filename = f"{base_filename}.png"
        else:
            filename = f"{base_filename}_sample_{i + 1:03d}.png"
        im.save(os.path.join(save_dir, filename))


def class_dir_from_image_path(rel_path: str) -> str:
    return os.path.basename(os.path.dirname(rel_path))


def build_arg_parser(description: str, default_out_dir: str):
    import argparse

    p = argparse.ArgumentParser(description)
    p.add_argument("--base_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--vae_model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--ip_ckpt", required=True, help="Path to ip_adapter.bin or ip_adapter.safetensors")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=3.5, help="guidance scale")
    p.add_argument("--scale", type=float, default=1.0, help="scale")
    p.add_argument("--num_samples", type=int, default=10, help="images per prompt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=default_out_dir)
    p.add_argument("--num_tokens", type=int, default=4, help="must match training")
    p.add_argument("--prompt", type=str, default=None, help="Prompt for image generation (optional).")
    p.add_argument("--taxonomic_prompt", action="store_true", help="If set, use taxonomic name as prompt.")
    p.add_argument("--biocap_caption", action="store_true", help="If set, use BioCap caption as prompt.")
    p.add_argument("--levels", type=int, default=7, help="Taxonomic levels to use (1=kingdom,...7=species).")
    p.add_argument("--json_file", "--json", dest="json_file", required=True, help="Path to JSON list of dicts.")
    p.add_argument("--dataset", type=str, default="inat", choices=["inat", "fishnet"], help="Dataset type.")
    p.add_argument(
        "--sae_dir",
        type=str,
        default=None,
        help="Directory containing both config.json and sae.pt.",
    )
    p.add_argument(
        "--sae_ckpt",
        type=str,
        default="/scratch/bio_diffusion/ip-adapter_runs/bioclip_sae/sae_model/sae.pt",
        help="Path to SAE checkpoint.",
    )
    p.add_argument(
        "--sae_alpha",
        type=float,
        default=0.5,
        help="Blending strength: 0=pure SAE path, 1=pure original BioCLIP.",
    )
    return p


def load_unique_items(json_file: str):
    with open(json_file, "r") as f:
        items = json.load(f)

    seen = set()
    unique_items = []
    for ex in items:
        tax = ex["taxonomic_name"]
        if tax not in seen:
            seen.add(tax)
            unique_items.append(ex)
    return unique_items, len(items)


def resolve_prompt(args, taxa_name: str, biocap_caption: str):
    if args.taxonomic_prompt:
        return "best quality, high quality photo of " + taxa_name
    if args.biocap_caption:
        return "best quality, high quality photo of " + biocap_caption
    return args.prompt


def write_debug_stats(out_dir: str, stats: dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "embedding_debug.json"), "w") as f:
        json.dump(stats, f, indent=2)


def build_sae_warning(sae_metadata: Dict, num_text_layers: int) -> Optional[str]:
    train_layer = sae_metadata.get("train_layer")
    tokens_mode = sae_metadata.get("tokens_mode")
    if tokens_mode == "content" and train_layer == num_text_layers - 1:
        return (
            f"SAE checkpoint was trained on layer {train_layer} content tokens, but BioCLIP pools from the "
            "EOT embedding. Because there is no later text block left to propagate content-token changes into EOT, "
            "this run is expected to stay close to vanilla BioCLIP in the final pooled embedding."
        )
    return None


@torch.inference_mode()
def run_sae_inference(args, wrapper_cls: Type[torch.nn.Module], mode_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    pipe = make_pipe(args.base_model, args.vae_model, device)

    base_bioclip, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    base_bioclip = base_bioclip.eval()
    bioclip_tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    if args.sae_dir:
        sae, sae_metadata = load_sae_from_dir(args.sae_dir, device="cpu")
    else:
        sae = load_sae(args.sae_ckpt, device="cpu")
        sae_metadata = {
            "sae_dir": None,
            "config_path": None,
            "sae_path": args.sae_ckpt,
            "train_layer": None,
            "tokens_mode": None,
        }

    wrapper_kwargs = {"alpha": args.sae_alpha}
    if sae_metadata.get("train_layer") is not None:
        wrapper_kwargs["layer_index"] = sae_metadata["train_layer"]
    wrapped_bioclip = wrapper_cls(base_bioclip, sae, **wrapper_kwargs).eval()
    num_text_layers = len(base_bioclip.transformer.resblocks)
    warning_message = build_sae_warning(sae_metadata, num_text_layers)

    config = PretrainedConfig.from_pretrained("MVRL/taxabind-config")
    taxabind = TaxaBind(config)
    location_encoder = taxabind.get_location_encoder().eval()
    taxabind_image_text_model = taxabind.get_image_text_encoder().eval()

    ip_model = IPAdapter(
        pipe,
        image_encoder_path=None,
        ip_ckpt=args.ip_ckpt,
        device=device,
        num_tokens=args.num_tokens,
        model_type="bioclip",
        bioclip=wrapped_bioclip,
        taxabind=taxabind_image_text_model,
        location_encoder=location_encoder,
    )

    unique_items, total_rows = load_unique_items(args.json_file)
    debug_taxa = []
    for entry in unique_items:
        taxa_name = entry["taxonomic_name"]
        if args.levels < 7:
            taxa_name = " ".join(taxa_name.split(" ")[: args.levels])
        debug_taxa.append(taxa_name)
    debug_stats = compute_embedding_debug_stats(
        ip_model.bioclip.bioclip,
        ip_model.bioclip,
        bioclip_tok,
        debug_taxa,
        device,
    )
    debug_stats["script_mode"] = mode_name
    debug_stats["sae_checkpoint"] = sae_metadata.get("sae_path", args.sae_ckpt)
    debug_stats["sae_directory"] = sae_metadata.get("sae_dir")
    debug_stats["sae_train_layer"] = sae_metadata.get("train_layer")
    debug_stats["sae_tokens_mode"] = sae_metadata.get("tokens_mode")
    debug_stats["expected_final_embedding_noop"] = warning_message is not None
    if warning_message is not None:
        debug_stats["warning"] = warning_message
    write_debug_stats(args.out_dir, debug_stats)

    print(f"Found {len(unique_items)} unique taxonomic names (from {total_rows} rows).")
    print(f"SAE mode: {mode_name}; alpha={args.sae_alpha}")
    if sae_metadata.get("sae_dir"):
        print(
            f"Using SAE directory {sae_metadata['sae_dir']} "
            f"(layer={sae_metadata.get('train_layer')}, tokens={sae_metadata.get('tokens_mode')})"
        )
    if warning_message is not None:
        print(f"WARNING: {warning_message}")

    for entry in tqdm(unique_items, total=len(unique_items)):
        taxa_name = entry["taxonomic_name"]
        biocap_caption = entry.get("text", "")

        if args.levels < 7:
            taxa_name = " ".join(taxa_name.split(" ")[: args.levels])

        if args.dataset == "fishnet":
            folder_name = taxa_name.replace(" ", "_").replace("/", "-")
            save_dir = os.path.join(args.out_dir, folder_name)
        else:
            rel_img_path = entry["image_file"]
            class_dir = class_dir_from_image_path(rel_img_path)
            save_dir = os.path.join(args.out_dir, class_dir)

        tokens = bioclip_tok(taxa_name).to(device)
        prompt = resolve_prompt(args, taxa_name, biocap_caption)

        images = ip_model.generate(
            pil_image=tokens,
            num_samples=args.num_samples,
            num_inference_steps=args.steps,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            prompt=prompt,
            scale=args.scale,
        )

        if args.dataset == "fishnet":
            save_images_fishnet(images, save_dir, taxa_name.replace("/", "-"), args.num_samples)
        else:
            save_images_per_class(images, save_dir, class_dir)

        print(f"    Saved {len(images)} images -> {save_dir}")

    print("Done.")
