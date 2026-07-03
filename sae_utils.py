"""
Shared SAE helpers for BioCLIP-based IP-Adapter inference.

This module provides:
  * lightweight SAE checkpoint loading compatible with `saev`
  * checkpoint-directory metadata loading (`config.json` + `sae.pt`)
  * a pooled-embedding SAE wrapper for A/B testing
  * a layer-aware SAE wrapper that reconstructs token states and then
    continues through the remainder of the BioCLIP text tower
  * lightweight embedding debug stats for script-side logging
"""

import io
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.model import text_global_pool


class MinimalSAE(nn.Module):
    """
    Sparse autoencoder with official-style encode/decode semantics.

    The checkpoint stores:
      W_enc: (d_model, d_sae)
      b_enc: (d_sae,)
      W_dec: (d_sae, d_model)
      b_dec: (d_model,)
    """

    def __init__(self, d_model: int, d_sae: int):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, d_model) -> (batch, d_sae) sparse latents."""
        return F.relu(x @ self.W_enc + self.b_enc)

    def decode(self, f_x: torch.Tensor) -> torch.Tensor:
        """(batch, d_sae) -> (batch, d_model) reconstructed vectors."""
        return f_x @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        """Returns (x_hat, f_x)."""
        f_x = self.encode(x)
        return self.decode(f_x), f_x


def load_sae(fpath: str, device: str = "cpu") -> MinimalSAE:
    """
    Load a `saev` sparse autoencoder checkpoint.

    File format: JSON header on the first line + a binary torch state_dict.
    """
    with open(fpath, "rb") as fd:
        header = json.loads(fd.readline())
        buffer = io.BytesIO(fd.read())

    cfg = header["cfg"]
    d_model, d_sae = cfg["d_model"], cfg["d_sae"]
    print(f"[SAE] d_model={d_model}, d_sae={d_sae}, activation={cfg['activation']['cls']}")

    sae = MinimalSAE(d_model, d_sae)
    sae.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
    return sae.to(device).eval()


def load_sae_metadata(sae_dir: str) -> Dict:
    """
    Load SAE directory metadata from the sibling config and checkpoint paths.
    """
    resolved_dir = os.path.abspath(sae_dir)
    config_path = os.path.join(resolved_dir, "config.json")
    sae_path = os.path.join(resolved_dir, "sae.pt")

    if not os.path.isdir(resolved_dir):
        raise FileNotFoundError(f"SAE directory not found: {resolved_dir}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing SAE config: {config_path}")
    if not os.path.isfile(sae_path):
        raise FileNotFoundError(f"Missing SAE checkpoint: {sae_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    train_data = config.get("train_data", {})
    return {
        "sae_dir": resolved_dir,
        "config_path": config_path,
        "sae_path": sae_path,
        "config": config,
        "train_layer": train_data.get("layer"),
        "tokens_mode": train_data.get("tokens"),
    }


def load_sae_from_dir(sae_dir: str, device: str = "cpu") -> Tuple[MinimalSAE, Dict]:
    """
    Load an SAE from a directory that contains `config.json` and `sae.pt`.
    """
    metadata = load_sae_metadata(sae_dir)
    sae = load_sae(metadata["sae_path"], device=device)
    return sae, metadata


def _rescale_to_match_norm(reconstructed: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """
    Match the reconstruction norm to the reference norm along the last dimension.

    This keeps the SAE intervention direction while reducing scale explosions,
    which is especially helpful for A/B generation comparisons.
    """
    reference_norm = reference.norm(dim=-1, keepdim=True)
    return F.normalize(reconstructed, dim=-1) * reference_norm


class _BioCLIPWithSAEBase(nn.Module):
    """Common functionality for pooled and layer-0 SAE wrappers."""

    mode_name = "base"

    def __init__(self, bioclip_model: nn.Module, sae: MinimalSAE, alpha: float = 0.0, preserve_scale: bool = True):
        super().__init__()
        self.bioclip = bioclip_model
        self.sae = sae
        self.alpha = alpha
        self.preserve_scale = preserve_scale

    def to(self, *args, **kwargs):
        """
        Keep the wrapped BioCLIP in the requested dtype/device, but retain the SAE
        in float32 for numerical stability.
        """
        super().to(*args, **kwargs)
        self.sae.float()
        return self

    def _run_sae(self, x: torch.Tensor):
        sae_input = x.float().to(device=self.sae.W_enc.device, dtype=self.sae.W_enc.dtype)
        f_x = self.sae.encode(sae_input)
        x_hat = self.sae.decode(f_x)
        return x_hat.float(), f_x.float()

    def _blend_reconstruction(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        original_float = original.float()
        reconstructed_float = reconstructed.float()
        if self.preserve_scale:
            reconstructed_float = _rescale_to_match_norm(reconstructed_float, original_float)
        return self.alpha * original_float + (1.0 - self.alpha) * reconstructed_float

    def _get_attn_mask(self, device: torch.device):
        attn_mask = getattr(self.bioclip, "attn_mask", None)
        if attn_mask is not None and attn_mask.device != device:
            attn_mask = attn_mask.to(device)
        return attn_mask


class BioCLIPWithSAEPooled(_BioCLIPWithSAEBase):
    """
    Apply the SAE directly to the final pooled BioCLIP embedding.

    This is the simpler exploratory baseline for A/B testing.
    """

    mode_name = "pooled"

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.bioclip.encode_text(tokens)
        if self.alpha >= 1.0:
            return x

        x_hat, _ = self._run_sae(x)
        out = self._blend_reconstruction(x, x_hat)
        return out.to(x.dtype)


class BioCLIPWithSAELayer(_BioCLIPWithSAEBase):
    """
    Apply the SAE to one BioCLIP text layer's content-token states, then
    continue the remaining text tower to recover the final pooled embedding.
    """

    mode_name = "layer"

    def __init__(
        self,
        bioclip_model: nn.Module,
        sae: MinimalSAE,
        alpha: float = 0.0,
        preserve_scale: bool = True,
        layer_index: int = 0,
    ):
        super().__init__(bioclip_model, sae, alpha=alpha, preserve_scale=preserve_scale)
        self.layer_index = layer_index

    @staticmethod
    def _content_mask(tokens: torch.Tensor) -> torch.Tensor:
        """
        Select content tokens only: positions strictly between the start token
        and the EOT token. Padding and special tokens are excluded.
        """
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        eot_indices = tokens.argmax(dim=-1, keepdim=True)
        return (positions > 0) & (positions < eot_indices)

    def _apply_sae_to_content_tokens(self, hidden_states: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        mask = self._content_mask(tokens)
        if self.alpha >= 1.0 or not mask.any():
            return hidden_states

        original_dtype = hidden_states.dtype
        content_states = hidden_states[mask]
        reconstructed, _ = self._run_sae(content_states)
        blended = self._blend_reconstruction(content_states, reconstructed).to(original_dtype)

        updated_states = hidden_states.clone()
        updated_states[mask] = blended
        return updated_states

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.alpha >= 1.0:
            return self.bioclip.encode_text(tokens)

        cast_dtype = self.bioclip.transformer.get_cast_dtype()
        x = self.bioclip.token_embedding(tokens).to(cast_dtype)
        x = x + self.bioclip.positional_embedding.to(cast_dtype)

        transformer = self.bioclip.transformer
        attn_mask = self._get_attn_mask(tokens.device)
        batch_first = getattr(transformer, "batch_first", False)
        if not batch_first:
            x = x.transpose(0, 1).contiguous()

        for block_idx, block in enumerate(transformer.resblocks):
            x = block(x, attn_mask=attn_mask)
            if block_idx == self.layer_index:
                if not batch_first:
                    x_batch = x.transpose(0, 1).contiguous()
                else:
                    x_batch = x

                x_batch = self._apply_sae_to_content_tokens(x_batch, tokens)

                if not batch_first:
                    x = x_batch.transpose(0, 1).contiguous()
                else:
                    x = x_batch

        if not batch_first:
            x = x.transpose(0, 1)

        x = self.bioclip.ln_final(x)
        x = text_global_pool(x, tokens, self.bioclip.text_pool_type)
        if self.bioclip.text_projection is not None:
            if isinstance(self.bioclip.text_projection, nn.Linear):
                x = self.bioclip.text_projection(x)
            else:
                x = x @ self.bioclip.text_projection
        return x


class BioCLIPLayerSAEAnalyzer(_BioCLIPWithSAEBase):
    """
    Helper for SAE feature analysis at a specific BioCLIP text layer.

    This exposes the exact intermediate states needed for latent inspection:
      * hidden states right after the target transformer block
      * flattened content-token activations
      * SAE latents and reconstructions
      * completion of the remaining text tower after optional latent edits
    """

    mode_name = "analysis"

    def __init__(
        self,
        bioclip_model: nn.Module,
        sae: MinimalSAE,
        layer_index: int = 0,
        preserve_scale: bool = False,
    ):
        super().__init__(bioclip_model, sae, alpha=0.0, preserve_scale=preserve_scale)
        self.layer_index = layer_index

    @staticmethod
    def content_mask(tokens: torch.Tensor) -> torch.Tensor:
        return BioCLIPWithSAELayer._content_mask(tokens)

    def _prepare_inputs(self, tokens: torch.Tensor):
        cast_dtype = self.bioclip.transformer.get_cast_dtype()
        x = self.bioclip.token_embedding(tokens).to(cast_dtype)
        x = x + self.bioclip.positional_embedding.to(cast_dtype)
        transformer = self.bioclip.transformer
        attn_mask = self._get_attn_mask(tokens.device)
        batch_first = getattr(transformer, "batch_first", False)
        if not batch_first:
            x = x.transpose(0, 1).contiguous()
        return x, transformer, attn_mask, batch_first

    def extract_layer_hidden_states(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Return batch-first hidden states immediately after `layer_index`.
        """
        x, transformer, attn_mask, batch_first = self._prepare_inputs(tokens)

        for block_idx, block in enumerate(transformer.resblocks):
            x = block(x, attn_mask=attn_mask)
            if block_idx == self.layer_index:
                if batch_first:
                    return x
                return x.transpose(0, 1).contiguous()

        raise ValueError(f"Layer index {self.layer_index} is out of range for this BioCLIP text tower.")

    def finalize_hidden_states(self, tokens: torch.Tensor, layer_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Run the remaining BioCLIP text blocks, pooling, and projection.
        """
        transformer = self.bioclip.transformer
        attn_mask = self._get_attn_mask(tokens.device)
        batch_first = getattr(transformer, "batch_first", False)
        x = layer_hidden_states
        if not batch_first:
            x = x.transpose(0, 1).contiguous()

        for block_idx in range(self.layer_index + 1, len(transformer.resblocks)):
            x = transformer.resblocks[block_idx](x, attn_mask=attn_mask)

        if not batch_first:
            x = x.transpose(0, 1).contiguous()

        x = self.bioclip.ln_final(x)
        x = text_global_pool(x, tokens, self.bioclip.text_pool_type)
        if self.bioclip.text_projection is not None:
            if isinstance(self.bioclip.text_projection, nn.Linear):
                x = self.bioclip.text_projection(x)
            else:
                x = x @ self.bioclip.text_projection
        return x

    def encode_content_latents(
        self,
        tokens: torch.Tensor,
        layer_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract flattened content-token states and run the SAE on them.
        """
        if layer_hidden_states is None:
            layer_hidden_states = self.extract_layer_hidden_states(tokens)

        content_mask = self.content_mask(tokens)
        content_indices = content_mask.nonzero(as_tuple=False)
        content_states = layer_hidden_states[content_mask]
        reconstructed_states, latents = self._run_sae(content_states)

        reconstructed_hidden_states = layer_hidden_states.clone()
        reconstructed_hidden_states[content_mask] = reconstructed_states.to(layer_hidden_states.dtype)

        baseline_final = self.finalize_hidden_states(tokens, layer_hidden_states)
        reconstructed_final = self.finalize_hidden_states(tokens, reconstructed_hidden_states)

        return {
            "layer_hidden_states": layer_hidden_states,
            "content_mask": content_mask,
            "content_indices": content_indices,
            "content_states": content_states,
            "latents": latents,
            "reconstructed_states": reconstructed_states,
            "reconstructed_hidden_states": reconstructed_hidden_states,
            "baseline_final_embeddings": baseline_final,
            "reconstructed_final_embeddings": reconstructed_final,
        }

    def decode_latents_to_hidden_states(
        self,
        layer_hidden_states: torch.Tensor,
        content_mask: torch.Tensor,
        content_states: torch.Tensor,
        latents: torch.Tensor,
        preserve_scale: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode edited latents back into full sequence hidden states.
        """
        preserve_scale = self.preserve_scale if preserve_scale is None else preserve_scale
        decoded_states = self.sae.decode(latents.float().to(self.sae.W_dec.dtype)).float()
        if preserve_scale:
            decoded_states = _rescale_to_match_norm(decoded_states, content_states.float())

        updated_hidden_states = layer_hidden_states.clone()
        updated_hidden_states[content_mask] = decoded_states.to(layer_hidden_states.dtype)
        return {
            "decoded_states": decoded_states,
            "updated_hidden_states": updated_hidden_states,
        }

    @staticmethod
    def apply_feature_edit(
        latents: torch.Tensor,
        feature_ids: Sequence[int],
        mode: str = "none",
        boost_factor: float = 2.0,
        boost_delta: float = 0.0,
    ) -> torch.Tensor:
        """
        Edit a common set of latent ids across all token positions.
        """
        edited = latents.clone()
        feature_ids = list(feature_ids)
        if not feature_ids or mode == "none":
            return edited

        if mode == "ablate":
            edited[:, feature_ids] = 0.0
        elif mode == "boost":
            if boost_factor != 1.0:
                edited[:, feature_ids] = edited[:, feature_ids] * boost_factor
            if boost_delta != 0.0:
                edited[:, feature_ids] = edited[:, feature_ids] + boost_delta
        else:
            raise ValueError(f"Unsupported edit mode: {mode}")
        return edited

    def reconstruct_with_edited_latents(
        self,
        tokens: torch.Tensor,
        layer_hidden_states: torch.Tensor,
        content_mask: torch.Tensor,
        content_states: torch.Tensor,
        latents: torch.Tensor,
        feature_ids: Sequence[int],
        mode: str = "none",
        boost_factor: float = 2.0,
        boost_delta: float = 0.0,
        preserve_scale: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        edited_latents = self.apply_feature_edit(
            latents,
            feature_ids=feature_ids,
            mode=mode,
            boost_factor=boost_factor,
            boost_delta=boost_delta,
        )
        decoded = self.decode_latents_to_hidden_states(
            layer_hidden_states=layer_hidden_states,
            content_mask=content_mask,
            content_states=content_states,
            latents=edited_latents,
            preserve_scale=preserve_scale,
        )
        final_embeddings = self.finalize_hidden_states(tokens, decoded["updated_hidden_states"])
        decoded["edited_latents"] = edited_latents
        decoded["final_embeddings"] = final_embeddings
        return decoded


@torch.inference_mode()
def compute_embedding_debug_stats(
    baseline_bioclip: nn.Module,
    wrapped_bioclip: nn.Module,
    tokenizer,
    taxa_texts: Sequence[str],
    device: str,
    max_items: int = 5,
) -> Dict:
    """
    Compare wrapped embeddings against vanilla BioCLIP for a small set of prompts.
    """
    sampled_texts: List[str] = []
    seen = set()
    for text in taxa_texts:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            sampled_texts.append(text)
        if len(sampled_texts) >= max_items:
            break

    if not sampled_texts:
        return {
            "mode": getattr(wrapped_bioclip, "mode_name", wrapped_bioclip.__class__.__name__),
            "sample_count": 0,
            "embedding_shape": None,
            "samples": [],
        }

    tokens = tokenizer(sampled_texts).to(device)
    baseline_embeds = baseline_bioclip.encode_text(tokens).float()
    wrapped_embeds = wrapped_bioclip.encode_text(tokens).float()
    cosine = F.cosine_similarity(wrapped_embeds, baseline_embeds, dim=-1)
    base_norms = baseline_embeds.norm(dim=-1)
    wrapped_norms = wrapped_embeds.norm(dim=-1)

    return {
        "mode": getattr(wrapped_bioclip, "mode_name", wrapped_bioclip.__class__.__name__),
        "alpha": float(getattr(wrapped_bioclip, "alpha", 0.0)),
        "preserve_scale": bool(getattr(wrapped_bioclip, "preserve_scale", False)),
        "sample_count": len(sampled_texts),
        "embedding_shape": list(wrapped_embeds.shape),
        "mean_cosine_similarity": float(cosine.mean().item()),
        "mean_baseline_norm": float(base_norms.mean().item()),
        "mean_wrapped_norm": float(wrapped_norms.mean().item()),
        "samples": [
            {
                "text": text,
                "baseline_norm": float(base_norms[idx].item()),
                "wrapped_norm": float(wrapped_norms[idx].item()),
                "cosine_similarity": float(cosine[idx].item()),
            }
            for idx, text in enumerate(sampled_texts)
        ],
    }


# Backwards-compatible aliases.
BioCLIPWithSAELayer0 = BioCLIPWithSAELayer
BioCLIPWithSAE = BioCLIPWithSAEPooled
