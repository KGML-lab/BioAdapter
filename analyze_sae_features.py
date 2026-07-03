import argparse
import csv
import json
import math
import os
from collections import defaultdict
from heapq import heappush, heapreplace
from itertools import count
from typing import Dict, List, Optional, Sequence

import open_clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sae_utils import BioCLIPLayerSAEAnalyzer, load_sae_from_dir


HISTOGRAM_EDGES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf")]
DEFAULT_SAE_DIR = "/scratch/bio_diffusion/ip-adapter_runs/bioclip_sae/sae_model/layer10"
DEFAULT_AUTO_INTERVENTION_FEATURES = 128


def parse_args():
    parser = argparse.ArgumentParser("Analyze layer-10 BioCLIP SAE features.")
    parser.add_argument("--sae_dir", default=DEFAULT_SAE_DIR, help="Directory containing config.json and sae.pt.")
    parser.add_argument("--json_file", required=True, help="Path to JSON list with taxonomic_name entries.")
    parser.add_argument("--out_dir", required=True, help="Directory for analysis outputs.")
    parser.add_argument("--levels", type=int, default=7, help="Taxonomic levels to keep (1=kingdom,...7=species).")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_texts", type=int, default=None, help="Optional cap on unique texts to analyze.")
    parser.add_argument("--top_k_examples", type=int, default=5)
    parser.add_argument("--top_k_tokens", type=int, default=5)
    parser.add_argument("--feature_ids", nargs="*", type=int, default=None, help="Optional latent ids for targeted intervention recomputation.")
    parser.add_argument("--boost_factor", type=float, default=2.0, help="Multiplicative boost for latent interventions.")
    parser.add_argument("--boost_delta", type=float, default=0.0, help="Additive boost for latent interventions.")
    parser.add_argument("--device", default=None, help="Override device; defaults to cuda if available.")
    parser.add_argument("--validate_with_saev", action="store_true", help="If saev is installed, compare local encode/decode with official loading semantics.")
    return parser.parse_args()


def truncate_taxonomic_name(text: str, levels: int) -> str:
    if levels >= 7:
        return text
    return " ".join(text.split(" ")[:levels])


def load_taxonomy_texts(json_file: str, levels: int, max_texts: Optional[int]) -> List[str]:
    with open(json_file, "r") as f:
        items = json.load(f)

    seen = set()
    texts = []
    for entry in items:
        tax = truncate_taxonomic_name(entry["taxonomic_name"], levels).strip()
        if not tax or tax in seen:
            continue
        seen.add(tax)
        texts.append(tax)
        if max_texts is not None and len(texts) >= max_texts:
            break
    return texts


def chunked(items: Sequence[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def update_topk_heap(heap, score: float, payload: Dict, limit: int, ticket_source) -> None:
    item = (float(score), next(ticket_source), payload)
    if len(heap) < limit:
        heappush(heap, item)
    elif score > heap[0][0]:
        heapreplace(heap, item)


def sorted_heap_payloads(heap) -> List[Dict]:
    return [item[2] for item in sorted(heap, key=lambda x: x[0], reverse=True)]


def classify_feature(firing_rate: float, token_count: int, example_count: int, max_activation: float) -> str:
    if token_count == 0:
        return "dead"
    if firing_rate >= 0.20:
        return "dense"
    if token_count < 5 or example_count < 3:
        return "rare"
    if firing_rate <= 0.05 and max_activation >= 1.0:
        return "interpretable_candidate"
    return "active"


def latent_score(mean_activation: float, example_count: int, firing_rate: float) -> float:
    if example_count == 0:
        return 0.0
    return mean_activation * math.log1p(example_count) / max(firing_rate, 1e-8)


def format_histogram_edges(edges: Sequence[float]) -> List[str]:
    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        if math.isinf(right):
            labels.append(f">={left:g}")
        else:
            labels.append(f"{left:g}-{right:g}")
    return labels


def decode_token_string(token_id: int, cache: Dict[int, str]) -> str:
    if token_id not in cache:
        text = open_clip.tokenizer.decode(torch.tensor([token_id]))
        text = text.replace("<start_of_text>", "[SOT]")
        text = text.replace("<end_of_text>", "[EOT]")
        text = text.replace("\n", " ").strip()
        cache[token_id] = text or f"id:{token_id}"
    return cache[token_id]


def write_json(path: str, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: str, rows: Sequence[Dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def nearest_neighbor(embedding: torch.Tensor, baseline_embeddings_normed: torch.Tensor, exclude_index: int) -> Dict[str, Optional[float]]:
    if baseline_embeddings_normed.shape[0] <= 1:
        return {
            "nearest_neighbor_index": None,
            "nearest_neighbor_score": None,
        }

    normed = F.normalize(embedding.unsqueeze(0), dim=-1)
    sims = (normed @ baseline_embeddings_normed.T).squeeze(0)
    sims[exclude_index] = -1e9
    score, index = sims.max(dim=0)
    return {
        "nearest_neighbor_index": int(index.item()),
        "nearest_neighbor_score": float(score.item()),
    }


def maybe_validate_with_saev(args, analyzer, tokenizer, texts: Sequence[str], device: str, sae_path: str) -> Dict:
    report = {
        "requested": bool(args.validate_with_saev),
        "available": False,
    }
    if not args.validate_with_saev:
        return report

    try:
        import saev.nn  # type: ignore
    except ImportError as exc:
        report["error"] = f"saev unavailable: {exc}"
        return report

    official_sae = saev.nn.load(sae_path, device="cpu")
    official_sae.eval()
    report["available"] = True

    sample_texts = list(texts[:2])
    if not sample_texts:
        report["error"] = "No texts available for validation."
        return report

    with torch.inference_mode():
        tokens = tokenizer(sample_texts).to(device)
        analysis = analyzer.encode_content_latents(tokens)
        local_latents = analysis["latents"].cpu()
        local_recon = analysis["reconstructed_states"].cpu()
        official_encode = official_sae.encode(analysis["content_states"].cpu())
        official_latents = official_encode.f_x if hasattr(official_encode, "f_x") else official_encode
        official_recon = official_sae.decode(official_latents)
        if official_recon.ndim == 3:
            official_recon = official_recon[:, -1, :]
        report["max_abs_latent_diff"] = float((local_latents - official_latents).abs().max().item())
        report["mean_abs_latent_diff"] = float((local_latents - official_latents).abs().mean().item())
        report["max_abs_recon_diff"] = float((local_recon - official_recon).abs().max().item())
        report["mean_abs_recon_diff"] = float((local_recon - official_recon).abs().mean().item())
    return report


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    feature_cards_dir = os.path.join(args.out_dir, "feature_cards")
    os.makedirs(feature_cards_dir, exist_ok=True)

    texts = load_taxonomy_texts(args.json_file, levels=args.levels, max_texts=args.max_texts)
    if not texts:
        raise ValueError("No valid taxonomic_name entries found in the provided JSON.")

    sae, sae_metadata = load_sae_from_dir(args.sae_dir, device=device)
    train_layer = sae_metadata.get("train_layer")
    if train_layer != 10:
        print(f"WARNING: expected layer10 SAE, but config says layer={train_layer}. Continuing with config value.")

    bioclip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    bioclip_model = bioclip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")
    analyzer = BioCLIPLayerSAEAnalyzer(
        bioclip_model,
        sae,
        layer_index=train_layer if train_layer is not None else 10,
        preserve_scale=False,
    ).to(device).eval()

    d_sae = sae.W_dec.shape[0]
    n_bins = len(HISTOGRAM_EDGES) - 1
    histogram_edges = torch.tensor(HISTOGRAM_EDGES, dtype=torch.float32)
    histogram_labels = format_histogram_edges(HISTOGRAM_EDGES)

    top_example_heaps = [[] for _ in range(d_sae)]
    top_token_heaps = [[] for _ in range(d_sae)]
    heap_ticket = count()
    token_string_cache: Dict[int, str] = {}

    total_content_tokens = 0
    token_count = torch.zeros(d_sae, dtype=torch.long)
    sum_activation = torch.zeros(d_sae, dtype=torch.float64)
    max_activation = torch.zeros(d_sae, dtype=torch.float32)
    example_count = torch.zeros(d_sae, dtype=torch.long)
    hist_counts = torch.zeros((d_sae, n_bins), dtype=torch.long)

    recon_state_cos_sum = 0.0
    recon_state_mse_sum = 0.0
    recon_state_count = 0
    recon_final_cos_sum = 0.0
    recon_final_count = 0

    baseline_embeddings = []

    print(f"Analyzing {len(texts)} unique taxonomy strings from {args.json_file}")
    print(f"Using SAE directory {sae_metadata['sae_dir']} (layer={sae_metadata.get('train_layer')}, tokens={sae_metadata.get('tokens_mode')})")
    print(f"Device: {device}")

    for start_idx, batch_texts in tqdm(list(chunked(texts, args.batch_size)), desc="Scanning corpus"):
        tokens = tokenizer(batch_texts).to(device)
        analysis = analyzer.encode_content_latents(tokens)

        content_indices = analysis["content_indices"].cpu()
        content_states = analysis["content_states"].cpu()
        latents = analysis["latents"].cpu()
        reconstructed_states = analysis["reconstructed_states"].cpu()
        baseline_final = analysis["baseline_final_embeddings"].float().cpu()
        reconstructed_final = analysis["reconstructed_final_embeddings"].float().cpu()
        token_ids = tokens.cpu()

        baseline_embeddings.append(baseline_final)

        if content_states.numel() > 0:
            recon_state_cos_sum += float(F.cosine_similarity(reconstructed_states, content_states, dim=-1).sum().item())
            recon_state_mse_sum += float(F.mse_loss(reconstructed_states, content_states, reduction="none").mean(dim=-1).sum().item())
            recon_state_count += int(content_states.shape[0])
            recon_final_cos_sum += float(F.cosine_similarity(reconstructed_final, baseline_final, dim=-1).sum().item())
            recon_final_count += int(baseline_final.shape[0])

        total_content_tokens += int(content_states.shape[0])
        token_count += (latents > 0).sum(dim=0).to(torch.long)
        sum_activation += latents.sum(dim=0).to(torch.float64)
        max_activation = torch.maximum(max_activation, latents.max(dim=0).values)

        positive_positions = (latents > 0).nonzero(as_tuple=False)
        if positive_positions.numel() > 0:
            positive_values = latents[positive_positions[:, 0], positive_positions[:, 1]]
            bin_indices = torch.bucketize(positive_values, histogram_edges[1:-1], right=False)
            flat_indices = positive_positions[:, 1] * n_bins + bin_indices
            counts = torch.bincount(flat_indices, minlength=d_sae * n_bins)
            hist_counts += counts.view(d_sae, n_bins)

        batch_assignments = content_indices[:, 0]
        batch_positions = content_indices[:, 1]

        for local_idx, text in enumerate(batch_texts):
            token_rows = (batch_assignments == local_idx).nonzero(as_tuple=False).flatten()
            if token_rows.numel() == 0:
                continue

            example_index = start_idx + local_idx
            example_latents = latents[token_rows]
            example_positions = batch_positions[token_rows]
            example_token_ids = token_ids[local_idx, example_positions]
            example_tokens = [decode_token_string(int(token_id), token_string_cache) for token_id in example_token_ids.tolist()]

            example_max, example_argmax = example_latents.max(dim=0)
            active_latents = (example_max > 0).nonzero(as_tuple=False).flatten()
            if active_latents.numel() > 0:
                example_count[active_latents] += 1

            for latent_id in active_latents.tolist():
                best_token_index = int(example_argmax[latent_id].item())
                best_score = float(example_max[latent_id].item())
                token_values = example_latents[:, latent_id].tolist()
                best_token_position = int(example_positions[best_token_index].item())
                best_token = example_tokens[best_token_index]
                example_payload = {
                    "latent_id": latent_id,
                    "example_index": example_index,
                    "text": text,
                    "activation": best_score,
                    "top_token": best_token,
                    "top_token_position": best_token_position,
                    "content_token_positions": [int(pos) for pos in example_positions.tolist()],
                    "content_tokens": example_tokens,
                    "token_values": [float(v) for v in token_values],
                }
                update_topk_heap(
                    top_example_heaps[latent_id],
                    score=best_score,
                    payload=example_payload,
                    limit=args.top_k_examples,
                    ticket_source=heap_ticket,
                )
                token_payload = {
                    "latent_id": latent_id,
                    "example_index": example_index,
                    "text": text,
                    "token": best_token,
                    "token_position": best_token_position,
                    "content_token_index": best_token_index,
                    "activation": best_score,
                }
                update_topk_heap(
                    top_token_heaps[latent_id],
                    score=best_score,
                    payload=token_payload,
                    limit=args.top_k_tokens,
                    ticket_source=heap_ticket,
                )

    baseline_embeddings = torch.cat(baseline_embeddings, dim=0)
    baseline_embeddings_normed = F.normalize(baseline_embeddings, dim=-1)
    if baseline_embeddings_normed.shape[0] > 1:
        baseline_similarity = baseline_embeddings_normed @ baseline_embeddings_normed.T
        baseline_similarity.fill_diagonal_(-1e9)
        baseline_nn_scores, baseline_nn_indices = baseline_similarity.max(dim=1)
    else:
        baseline_nn_scores = torch.tensor([])
        baseline_nn_indices = torch.tensor([], dtype=torch.long)

    feature_summary_rows = []
    top_examples_rows = []
    top_tokens_rows = []
    feature_cards = {}

    for latent_id in range(d_sae):
        active_token_count = int(token_count[latent_id].item())
        feature_example_count = int(example_count[latent_id].item())
        firing_rate = (active_token_count / total_content_tokens) if total_content_tokens > 0 else 0.0
        mean_activation = float((sum_activation[latent_id] / max(active_token_count, 1)).item())
        feature_max_activation = float(max_activation[latent_id].item())
        status = classify_feature(firing_rate, active_token_count, feature_example_count, feature_max_activation)
        score = latent_score(mean_activation, feature_example_count, firing_rate)

        top_examples = sorted_heap_payloads(top_example_heaps[latent_id])
        top_tokens = sorted_heap_payloads(top_token_heaps[latent_id])
        summary_row = {
            "latent_id": latent_id,
            "status": status,
            "token_count": active_token_count,
            "example_count": feature_example_count,
            "firing_rate": firing_rate,
            "mean_activation_when_active": mean_activation,
            "max_activation": feature_max_activation,
            "score": score,
        }
        feature_summary_rows.append(summary_row)
        top_examples_rows.append({"latent_id": latent_id, "top_examples": top_examples})
        top_tokens_rows.append({"latent_id": latent_id, "top_tokens": top_tokens})

        feature_cards[latent_id] = {
            "latent_id": latent_id,
            "summary": summary_row,
            "top_examples": top_examples,
            "top_tokens": top_tokens,
            "activation_histogram": {
                "bin_labels": histogram_labels,
                "counts": [int(x) for x in hist_counts[latent_id].tolist()],
            },
            "intervention": None,
        }

    if args.feature_ids:
        selected_feature_ids = {
            latent_id
            for latent_id in args.feature_ids
            if 0 <= latent_id < d_sae and feature_cards[latent_id]["top_examples"]
        }
    else:
        auto_candidates = [
            row for row in feature_summary_rows if row["token_count"] > 0 and feature_cards[row["latent_id"]]["top_examples"]
        ]
        auto_candidates.sort(key=lambda row: row["score"], reverse=True)
        selected_feature_ids = {
            int(row["latent_id"])
            for row in auto_candidates[:DEFAULT_AUTO_INTERVENTION_FEATURES]
        }

    representative_by_example = defaultdict(list)
    for latent_id in sorted(selected_feature_ids):
        representative_example = feature_cards[latent_id]["top_examples"][0]["example_index"]
        representative_by_example[representative_example].append(latent_id)

    print(f"Computing representative interventions for {len(selected_feature_ids)} latents...")
    for example_index, latent_ids in tqdm(sorted(representative_by_example.items()), desc="Interventions"):
        text = texts[example_index]
        tokens = tokenizer([text]).to(device)
        analysis = analyzer.encode_content_latents(tokens)
        baseline_embedding = analysis["baseline_final_embeddings"].float().cpu()[0]
        baseline_neighbor = nearest_neighbor(baseline_embedding, baseline_embeddings_normed, exclude_index=example_index)
        baseline_neighbor_index = baseline_neighbor["nearest_neighbor_index"]
        baseline_neighbor_text = texts[baseline_neighbor_index] if baseline_neighbor_index is not None else None

        for latent_id in latent_ids:
            intervention_report = {
                "representative_example_index": example_index,
                "representative_text": text,
                "baseline_nearest_neighbor_index": baseline_neighbor_index,
                "baseline_nearest_neighbor_text": baseline_neighbor_text,
                "baseline_nearest_neighbor_score": baseline_neighbor["nearest_neighbor_score"],
            }

            for mode in ("ablate", "boost"):
                edited = analyzer.reconstruct_with_edited_latents(
                    tokens=tokens,
                    layer_hidden_states=analysis["layer_hidden_states"],
                    content_mask=analysis["content_mask"],
                    content_states=analysis["content_states"],
                    latents=analysis["latents"],
                    feature_ids=[latent_id],
                    mode=mode,
                    boost_factor=args.boost_factor,
                    boost_delta=args.boost_delta,
                    preserve_scale=False,
                )
                edited_embedding = edited["final_embeddings"].float().cpu()[0]
                cosine = float(F.cosine_similarity(edited_embedding.unsqueeze(0), baseline_embedding.unsqueeze(0), dim=-1).item())
                neighbor = nearest_neighbor(edited_embedding, baseline_embeddings_normed, exclude_index=example_index)
                neighbor_index = neighbor["nearest_neighbor_index"]
                intervention_report[mode] = {
                    "cosine_to_baseline": cosine,
                    "cosine_shift": 1.0 - cosine,
                    "nearest_neighbor_index": neighbor_index,
                    "nearest_neighbor_text": texts[neighbor_index] if neighbor_index is not None else None,
                    "nearest_neighbor_score": neighbor["nearest_neighbor_score"],
                    "nearest_neighbor_changed": neighbor_index != baseline_neighbor_index,
                }

            feature_cards[latent_id]["intervention"] = intervention_report

    validation_report = maybe_validate_with_saev(
        args=args,
        analyzer=analyzer,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
        sae_path=sae_metadata["sae_path"],
    )

    summary_fieldnames = [
        "latent_id",
        "status",
        "token_count",
        "example_count",
        "firing_rate",
        "mean_activation_when_active",
        "max_activation",
        "score",
    ]
    write_csv(os.path.join(args.out_dir, "feature_summary.csv"), feature_summary_rows, summary_fieldnames)
    write_jsonl(os.path.join(args.out_dir, "top_examples.jsonl"), top_examples_rows)
    write_jsonl(os.path.join(args.out_dir, "top_tokens.jsonl"), top_tokens_rows)
    for latent_id, card in feature_cards.items():
        write_json(os.path.join(feature_cards_dir, f"latent_{latent_id:05d}.json"), card)

    analysis_metadata = {
        "sae_directory": sae_metadata["sae_dir"],
        "sae_checkpoint": sae_metadata["sae_path"],
        "sae_train_layer": sae_metadata.get("train_layer"),
        "sae_tokens_mode": sae_metadata.get("tokens_mode"),
        "json_file": os.path.abspath(args.json_file),
        "levels": args.levels,
        "batch_size": args.batch_size,
        "max_texts": args.max_texts,
        "top_k_examples": args.top_k_examples,
        "top_k_tokens": args.top_k_tokens,
        "requested_feature_ids": sorted(args.feature_ids) if args.feature_ids else [],
        "intervention_feature_ids": sorted(selected_feature_ids),
        "intervention_feature_count": len(selected_feature_ids),
        "device": device,
        "text_count": len(texts),
        "total_content_tokens": total_content_tokens,
    }
    write_json(os.path.join(args.out_dir, "analysis_metadata.json"), analysis_metadata)

    recon_metrics = {
        "layer_reconstruction_mean_cosine": (recon_state_cos_sum / recon_state_count) if recon_state_count else None,
        "layer_reconstruction_mean_mse": (recon_state_mse_sum / recon_state_count) if recon_state_count else None,
        "final_embedding_mean_cosine": (recon_final_cos_sum / recon_final_count) if recon_final_count else None,
        "final_embedding_mean_shift": (1.0 - (recon_final_cos_sum / recon_final_count)) if recon_final_count else None,
        "validation_with_saev": validation_report,
    }
    write_json(os.path.join(args.out_dir, "recon_metrics.json"), recon_metrics)

    print(f"Wrote analysis outputs to {args.out_dir}")
    print(f"Feature summary rows: {len(feature_summary_rows)}")
    print(f"Representative interventions computed for {len(selected_feature_ids)} latents")


if __name__ == "__main__":
    main()
