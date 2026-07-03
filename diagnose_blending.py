"""
Diagnose SAE reconstruction quality and test blending strategies.

Shows what the IP-Adapter actually receives under different SAE integration modes.
"""

import torch
import torch.nn.functional as F
import open_clip
from sae_utils import load_sae

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load models ──────────────────────────────────────────────────────
bioclip, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
bioclip = bioclip.to(device).eval()
tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

sae = load_sae("/scratch/bio_diffusion/ip-adapter_runs/bioclip_sae/sae_model/sae.pt", device=device)

species = [
    "Panthera leo",
    "Aquila chrysaetos",
    "Salvia officinalis",
    "Danaus plexippus",
    "Canis lupus",
    "Quercus robur",
    "Amanita muscaria",
    "Corvus corax",
]

tokens = tok(species).to(device)

with torch.no_grad():
    x = bioclip.encode_text(tokens)  # (8, 768), raw BioCLIP embeddings
    x_float = x.float()
    orig_norm = x_float.norm(dim=-1, keepdim=True)

    # SAE reconstruction
    x_hat, f_x = sae(x_float.to(sae.W_dec.dtype))
    x_hat = x_hat.float()

    # Different strategies for what to feed IP-Adapter
    # 1. Raw x_hat (what we had before norm fix)
    # 2. Norm-rescaled x_hat (current fix)
    x_hat_rescaled = F.normalize(x_hat, dim=-1) * orig_norm
    # 3. Blended at various alphas: x_final = alpha * x + (1-alpha) * x_hat_rescaled
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("=" * 100)
    print("RAW BIOCLIP vs SAE RECONSTRUCTION QUALITY")
    print("=" * 100)
    print(f"{'Species':<25} {'x norm':>8} {'x_hat norm':>10} {'rescaled':>10} {'cos_sim':>8} {'L0':>6}")
    print("-" * 100)

    for i, sp in enumerate(species):
        cos = F.cosine_similarity(x_float[i:i+1], x_hat[i:i+1]).item()
        l0 = (f_x[i] > 0).sum().item()
        print(f"{sp:<25} {orig_norm[i].item():8.2f} {x_hat[i].norm().item():10.2f} "
              f"{x_hat_rescaled[i].norm().item():10.2f} {cos:8.4f} {l0:6d}")

    print()
    print("=" * 100)
    print("BLENDING: x_final = alpha * x_original + (1 - alpha) * x_hat_rescaled")
    print("alpha=1.0 means PURE original BioCLIP (no SAE), alpha=0.0 means PURE SAE")
    print("=" * 100)
    print(f"{'Species':<25}", end="")
    for a in alphas:
        print(f"  cos@α={a:.2f}", end="")
    print(f"  {'norm@α=0.5':>10}")
    print("-" * 100)

    for i, sp in enumerate(species):
        print(f"{sp:<25}", end="")
        for a in alphas:
            blended = a * x_float[i] + (1 - a) * x_hat_rescaled[i]
            cos_with_orig = F.cosine_similarity(blended.unsqueeze(0), x_float[i:i+1]).item()
            print(f"  {cos_with_orig:10.4f}", end="")
        blended_05 = 0.5 * x_float[i] + 0.5 * x_hat_rescaled[i]
        print(f"  {blended_05.norm().item():10.2f}")

    # Show pairwise cos sims between species for different modes
    print()
    print("=" * 100)
    print("PAIRWISE SPECIES DISCRIMINATION (are species distinguishable?)")
    print("=" * 100)
    for mode_name, emb in [("Original BioCLIP", x_float),
                            ("SAE rescaled (α=0)", x_hat_rescaled),
                            ("Blended (α=0.5)", 0.5 * x_float + 0.5 * x_hat_rescaled),
                            ("Blended (α=0.75)", 0.75 * x_float + 0.25 * x_hat_rescaled)]:
        emb_norm = F.normalize(emb, dim=-1)
        sim_matrix = emb_norm @ emb_norm.T
        # Get mean off-diagonal similarity
        mask = ~torch.eye(len(species), dtype=bool, device=device)
        mean_sim = sim_matrix[mask].mean().item()
        min_sim = sim_matrix[mask].min().item()
        max_sim = sim_matrix[mask].max().item()
        print(f"  {mode_name:<30}  mean_pair_sim={mean_sim:.4f}  min={min_sim:.4f}  max={max_sim:.4f}")

    print()
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    avg_cos = F.cosine_similarity(x_float, x_hat).mean().item()
    print(f"  Average reconstruction cos_sim: {avg_cos:.4f}")
    if avg_cos < 0.7:
        print(f"  ⚠ Reconstruction quality is LOW (cos_sim < 0.7).")
        print(f"  → Pure SAE output (α=0) will look bad because direction differs too much from training.")
        print(f"  → Try blending: α=0.5 or α=0.75 to preserve most of the original signal.")
        print(f"  → Or: use SAE only for analysis/interpretability, not for generation.")
    else:
        print(f"  ✓ Reconstruction quality is reasonable (cos_sim >= 0.7).")
        print(f"  → Pure SAE output should work, but blending may still improve quality.")
