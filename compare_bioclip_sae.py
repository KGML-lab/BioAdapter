"""
Get SAE-reconstructed BioCLIP text embeddings.

Pipeline:
  taxonomic_name
      → BioCLIP tokenizer
      → BioCLIP.encode_text()   → x       shape: (1, 768), norm ≈ 23
      → L2-normalize            → x_norm  shape: (1, 768), norm = 1
      → SAE encoder (ReLU)      → f_x     shape: (1, 12288), sparse
      → SAE decoder             → x_hat   shape: (1, 768)   ← use this

SAE file format (saev library): JSON header (first line) + binary torch state_dict.
SAE weights: W_enc (768→12288), b_enc, W_dec (12288→768), b_dec.
"""

import io
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# ─────────────────────────────────────────────
# Minimal SAE loader (no saev dependency)
# ─────────────────────────────────────────────

class MinimalSAE(nn.Module):
    """Minimal sparse autoencoder: encoder (ReLU) + linear decoder."""

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_model) → f_x: (batch, d_sae)  [sparse]

        Uses W_dec.T instead of the stored W_enc for encoding.
        The saved W_enc from saev v0.1.0 is degenerate for BioCLIP inputs.
        """
        h_x = x @ self.W_dec.T + self.b_enc
        return F.relu(h_x)

    def decode(self, f_x: torch.Tensor) -> torch.Tensor:
        """f_x: (batch, d_sae) → x_hat: (batch, d_model)"""
        return f_x @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        """Returns (x_hat, f_x)."""
        f_x = self.encode(x)
        x_hat = self.decode(f_x)
        return x_hat, f_x


def load_sae(fpath: str, device: str = "cpu") -> MinimalSAE:
    """
    Load an SAE saved by the `saev` library.
    File format: JSON header on first line + binary torch state_dict.
    """
    with open(fpath, "rb") as fd:
        header = json.loads(fd.readline())
        buffer = io.BytesIO(fd.read())

    cfg = header["cfg"]
    d_model = cfg["d_model"]
    d_sae   = cfg["d_sae"]
    print(f"[SAE cfg]  d_model={d_model}, d_sae={d_sae}, "
          f"activation={cfg['activation']}")

    sae = MinimalSAE(d_model, d_sae)
    state_dict = torch.load(buffer, weights_only=True, map_location=device)
    print(f"[SAE keys] {list(state_dict.keys())}")
    sae.load_state_dict(state_dict)
    sae = sae.to(device).eval()
    return sae


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

SAE_PATH = "/scratch/bio_diffusion/ip-adapter_runs/bioclip_sae/sae_model/sae.pt"
BIOCLIP_CKPT = "hf-hub:imageomics/bioclip-2"

EXAMPLE_TAXA = [
    "Animalia Chordata Aves Passeriformes Fringillidae Spinus tristis",
    "Plantae Tracheophyta Magnoliopsida Lamiales Lamiaceae Salvia officinalis",
    "Animalia Chordata Mammalia Carnivora Felidae Panthera leo",
]


@torch.inference_mode()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # ── 1. Inspect the raw file header ────────────────────
    print("═" * 60)
    print("  STEP 1: Inspect sae.pt file header")
    print("═" * 60)
    with open(SAE_PATH, "rb") as fd:
        header_line = fd.readline()
        binary_start = fd.tell()
        fd.seek(0, 2)
        file_size = fd.tell()
    header = json.loads(header_line)
    print(f"  Header line length: {len(header_line)} bytes")
    print(f"  Binary data starts at byte: {binary_start}")
    print(f"  Binary data size: {file_size - binary_start} bytes")
    print(f"  Schema version: {header.get('schema', 'MISSING')}")
    print(f"  Header keys: {list(header.keys())}")
    # Print full header for debugging
    import pprint
    print("  Full header:")
    pprint.pprint(header, indent=4, width=100)

    # ── 2. Load SAE with our loader ───────────────────────
    print(f"\n{'═' * 60}")
    print("  STEP 2: Load SAE with MinimalSAE")
    print("═" * 60)
    sae = load_sae(SAE_PATH, device=device)
    print(f"  W_enc  shape={tuple(sae.W_enc.shape)}  dtype={sae.W_enc.dtype}")
    print(f"  b_enc  shape={tuple(sae.b_enc.shape)}  dtype={sae.b_enc.dtype}")
    print(f"  W_dec  shape={tuple(sae.W_dec.shape)}  dtype={sae.W_dec.dtype}")
    print(f"  b_dec  shape={tuple(sae.b_dec.shape)}  dtype={sae.b_dec.dtype}")
    print(f"  W_enc first 5 values: {sae.W_enc.data.flatten()[:5].tolist()}")
    print(f"  W_dec first 5 values: {sae.W_dec.data.flatten()[:5].tolist()}")
    print(f"  b_enc first 5 values: {sae.b_enc.data[:5].tolist()}")
    print(f"  b_dec first 5 values: {sae.b_dec.data[:5].tolist()}")

    # ── 3. Try loading with saev library directly ─────────
    print(f"\n{'═' * 60}")
    print("  STEP 3: Try loading with saev library (if installed)")
    print("═" * 60)
    try:
        import saev.nn
        sae_official = saev.nn.load(SAE_PATH, device=device)
        print(f"  ✓ saev library loaded successfully!")
        print(f"  Official W_enc shape: {tuple(sae_official.W_enc.shape)}")
        print(f"  Official W_dec shape: {tuple(sae_official.W_dec.shape)}")
        print(f"  Official W_enc first 5: {sae_official.W_enc.data.flatten()[:5].tolist()}")
        print(f"  Official W_dec first 5: {sae_official.W_dec.data.flatten()[:5].tolist()}")
        # Compare
        enc_match = torch.allclose(sae.W_enc.data, sae_official.W_enc.data, atol=1e-6)
        dec_match = torch.allclose(sae.W_dec.data, sae_official.W_dec.data, atol=1e-6)
        print(f"  W_enc matches: {enc_match}")
        print(f"  W_dec matches: {dec_match}")

        # Test with official forward
        print(f"\n  Testing official SAE encode on BioCLIP-2 embedding...")
        bioclip_model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_CKPT)
        bioclip_tok = open_clip.get_tokenizer(BIOCLIP_CKPT)
        bioclip_model = bioclip_model.to(device).eval()
        tokens = bioclip_tok(EXAMPLE_TAXA[0]).to(device)
        x = bioclip_model.encode_text(tokens).float()
        enc_out = sae_official.encode(x)
        print(f"  Official encode type: {type(enc_out)}")
        if hasattr(enc_out, 'f_x'):
            f_x = enc_out.f_x
            print(f"  Official L0: {(f_x > 0).sum().item()}/12288")
            print(f"  Official f_x max: {f_x.max().item():.4f}")
        elif isinstance(enc_out, torch.Tensor):
            print(f"  Official L0: {(enc_out > 0).sum().item()}/12288")
            print(f"  Official enc max: {enc_out.max().item():.4f}")
        else:
            print(f"  Official encode output: {enc_out}")

    except ImportError:
        print("  ✗ saev library not installed — skipping")
        print("    To install: pip install saev")

    # ── 4. Sanity check: random input ─────────────────────
    print(f"\n{'═' * 60}")
    print("  STEP 4: Random input sanity check")
    print("═" * 60)
    torch.manual_seed(42)
    x_rand = torch.randn(1, 768, device=device)
    # Test at various norms
    for norm_val in [1.0, 23.0, 100.0]:
        x_test = x_rand / x_rand.norm() * norm_val
        h_x = x_test @ sae.W_enc + sae.b_enc
        n_pos = (h_x > 0).sum().item()
        print(f"  random (norm={norm_val:5.1f}): h_x mean={h_x.mean():.4f}  "
              f"max={h_x.max():.4f}  positive={n_pos}/12288")

    # ── 5. Check if W_enc ≈ W_dec.T (initialized that way) ─
    print(f"\n{'═' * 60}")
    print("  STEP 5: W_enc vs W_dec.T relationship")
    print("═" * 60)
    # saev initializes W_enc = W_dec.T, training may diverge
    diff = (sae.W_enc.data - sae.W_dec.data.T).abs()
    print(f"  |W_enc - W_dec.T|: mean={diff.mean():.6f}  max={diff.max():.6f}")
    cos_sim = F.cosine_similarity(sae.W_enc.data.flatten().unsqueeze(0),
                                   sae.W_dec.data.T.flatten().unsqueeze(0))
    print(f"  cosine_similarity(W_enc, W_dec.T) = {cos_sim.item():.6f}")

    # ── 6. The real test: BioCLIP-2 embedding ─────────────
    print(f"\n{'═' * 60}")
    print("  STEP 6: BioCLIP-2 embeddings through SAE")
    print("═" * 60)

    if 'bioclip_model' not in dir():
        bioclip_model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_CKPT)
        bioclip_tok = open_clip.get_tokenizer(BIOCLIP_CKPT)
        bioclip_model = bioclip_model.to(device).eval()

    for taxa in EXAMPLE_TAXA:
        tokens = bioclip_tok(taxa).to(device)
        x = bioclip_model.encode_text(tokens).float()

        # Raw
        h_x_raw = x @ sae.W_enc + sae.b_enc
        f_x_raw = F.relu(h_x_raw)
        x_hat_raw = f_x_raw @ sae.W_dec + sae.b_dec

        # Also try: encode with W_dec.T instead of W_enc (transpose bug?)
        h_x_transposed = x @ sae.W_dec.T + sae.b_enc
        f_x_transposed = F.relu(h_x_transposed)

        species = taxa.split()[-1]
        print(f"  {species:20s}  norm={x.norm().item():.2f}  "
              f"W_enc→ L0={int((f_x_raw > 0).sum()):4d}  h_max={h_x_raw.max():.3f}  |  "
              f"W_dec.T→ L0={int((f_x_transposed > 0).sum()):4d}  h_max={h_x_transposed.max():.3f}")

    # ── 7. Check W_dec column norms (config has normalize_w_dec=true) ─
    print(f"\n{'═' * 60}")
    print("  STEP 7: W_dec row norms (config: normalize_w_dec=true)")
    print("═" * 60)
    # W_dec shape is (d_sae, d_model) = (12288, 768)
    # saev normalizes rows: W_dec[i,:] / ||W_dec[i,:]||
    row_norms = sae.W_dec.data.norm(dim=1)
    print(f"  W_dec row norms: mean={row_norms.mean():.4f}  min={row_norms.min():.4f}  "
          f"max={row_norms.max():.4f}  std={row_norms.std():.6f}")
    print(f"  Are rows unit-norm? {torch.allclose(row_norms, torch.ones_like(row_norms), atol=0.01)}")

    # W_enc column norms
    col_norms = sae.W_enc.data.norm(dim=0)
    print(f"  W_enc col norms: mean={col_norms.mean():.4f}  min={col_norms.min():.4f}  "
          f"max={col_norms.max():.4f}  std={col_norms.std():.6f}")

    print(f"\n{'═' * 60}")
    print("  DONE — check if W_dec.T gives L0>0 (→ transpose bug)")
    print("  or if saev official gives different results (→ loading bug)")
    print("═" * 60)

    # ── 8. End-to-end test with fixed encode (W_dec.T) ────
    print(f"\n{'═' * 60}")
    print("  STEP 8: End-to-end SAE with W_dec.T encoder (the fix)")
    print("═" * 60)

    if 'bioclip_model' not in dir():
        bioclip_model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_CKPT)
        bioclip_tok = open_clip.get_tokenizer(BIOCLIP_CKPT)
        bioclip_model = bioclip_model.to(device).eval()

    embeddings = {}
    for taxa in EXAMPLE_TAXA:
        tokens = bioclip_tok(taxa).to(device)
        x = bioclip_model.encode_text(tokens).float()
        x_hat, f_x = sae(x)  # now uses W_dec.T internally
        l0 = (f_x > 0).sum().item()
        species = taxa.split()[-1]
        embeddings[species] = x_hat
        print(f"  {species:20s}  L0={l0:5d}  x_hat_norm={x_hat.norm().item():.4f}  "
              f"recon_cos={F.cosine_similarity(x, x_hat).item():.4f}")

    # Pairwise cosine similarities between species
    species_list = list(embeddings.keys())
    print(f"\n  Pairwise x_hat cosine similarities:")
    for i, s1 in enumerate(species_list):
        for j, s2 in enumerate(species_list):
            if j > i:
                cos = F.cosine_similarity(embeddings[s1], embeddings[s2]).item()
                print(f"    {s1} vs {s2}: {cos:.4f}")


if __name__ == "__main__":
    main()
