import argparse
import csv
import glob
import html
import json
import os
from typing import Dict, List, Sequence


def parse_args():
    parser = argparse.ArgumentParser("Render static HTML feature cards for SAE analysis outputs.")
    parser.add_argument("--analysis_dir", required=True, help="Directory produced by analyze_sae_features.py.")
    parser.add_argument("--out_dir", default=None, help="HTML output directory. Defaults to <analysis_dir>/html.")
    return parser.parse_args()


def read_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def read_csv_rows(path: str) -> List[Dict]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def safe_float(value, default=0.0) -> float:
    if value in (None, "", "None"):
        return default
    return float(value)


def safe_int(value, default=0) -> int:
    if value in (None, "", "None"):
        return default
    return int(value)


def svg_bar_chart(labels: Sequence[str], values: Sequence[float], title: str, color: str = "#2563eb") -> str:
    width = 720
    height = 240
    margin_left = 40
    margin_bottom = 70
    chart_width = width - margin_left - 20
    chart_height = height - 30 - margin_bottom
    if not values:
        return "<p>No data available.</p>"

    max_value = max(values) if max(values) > 0 else 1.0
    bar_width = max(chart_width / max(len(values), 1) - 8, 10)
    gap = (chart_width - bar_width * len(values)) / max(len(values), 1)
    svg = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{margin_left}" y="20" font-size="14" font-weight="600">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{30 + chart_height}" x2="{width - 20}" y2="{30 + chart_height}" stroke="#555"/>',
        f'<line x1="{margin_left}" y1="30" x2="{margin_left}" y2="{30 + chart_height}" stroke="#555"/>',
    ]

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + gap / 2 + idx * (bar_width + gap)
        bar_height = 0 if max_value == 0 else (value / max_value) * chart_height
        y = 30 + chart_height - bar_height
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" rx="3"/>')
        svg.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{height - 42}" text-anchor="end" font-size="10" '
            f'transform="rotate(-35 {x + bar_width / 2:.1f},{height - 42})">{html.escape(str(label))}</text>'
        )
        svg.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-size="10">{value:.3g}</text>')

    svg.append("</svg>")
    return "".join(svg)


def render_token_sequence(tokens: Sequence[str], values: Sequence[float]) -> str:
    if not tokens:
        return "<p>No token data.</p>"
    max_value = max(values) if values else 0.0
    spans = []
    for token, value in zip(tokens, values):
        intensity = 0.0 if max_value <= 0 else min(value / max_value, 1.0)
        alpha = 0.08 + 0.55 * intensity
        color = f"rgba(37, 99, 235, {alpha:.3f})"
        text = html.escape(token)
        spans.append(
            f'<span class="token" style="background:{color}" title="activation={value:.4f}">{text}</span>'
        )
    return '<div class="token-seq">' + " ".join(spans) + "</div>"


def render_intervention_section(card: Dict) -> str:
    intervention = card.get("intervention")
    if not intervention:
        return "<p class='muted'>Intervention metrics were not computed for this latent.</p>"

    rows = []
    for mode in ("ablate", "boost"):
        payload = intervention.get(mode) or {}
        rows.append(
            "<tr>"
            f"<td>{mode}</td>"
            f"<td>{safe_float(payload.get('cosine_to_baseline')):.4f}</td>"
            f"<td>{safe_float(payload.get('cosine_shift')):.4f}</td>"
            f"<td>{html.escape(str(payload.get('nearest_neighbor_text')) if payload.get('nearest_neighbor_text') is not None else '-')}</td>"
            f"<td>{'yes' if payload.get('nearest_neighbor_changed') else 'no'}</td>"
            "</tr>"
        )

    shift_labels = ["ablate", "boost"]
    shift_values = [
        safe_float((intervention.get("ablate") or {}).get("cosine_shift")),
        safe_float((intervention.get("boost") or {}).get("cosine_shift")),
    ]

    table = (
        "<table class='metrics-table'><thead><tr>"
        "<th>edit</th><th>cosine to baseline</th><th>cosine shift</th><th>nearest neighbor</th><th>changed?</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    meta = (
        f"<p><strong>Representative text:</strong> {html.escape(intervention['representative_text'])}</p>"
        f"<p><strong>Baseline nearest neighbor:</strong> {html.escape(str(intervention.get('baseline_nearest_neighbor_text') or '-'))}</p>"
    )
    return meta + svg_bar_chart(shift_labels, shift_values, "Embedding cosine shift after edit", color="#dc2626") + table


def render_top_examples(card: Dict) -> str:
    blocks = []
    for idx, example in enumerate(card.get("top_examples", []), start=1):
        token_html = render_token_sequence(example.get("content_tokens", []), example.get("token_values", []))
        blocks.append(
            "<div class='example-card'>"
            f"<h3>Example {idx}</h3>"
            f"<p><strong>Text:</strong> {html.escape(example['text'])}</p>"
            f"<p><strong>Activation:</strong> {example['activation']:.4f} "
            f"| <strong>Top token:</strong> {html.escape(example['top_token'])} "
            f"| <strong>Seq pos:</strong> {example['top_token_position']}</p>"
            f"{token_html}"
            "</div>"
        )
    return "".join(blocks) if blocks else "<p class='muted'>No activating examples for this latent.</p>"


def render_top_tokens(card: Dict) -> str:
    top_tokens = card.get("top_tokens", [])
    if not top_tokens:
        return "<p class='muted'>No token-level activations recorded.</p>"
    labels = [entry["token"] for entry in top_tokens]
    values = [safe_float(entry["activation"]) for entry in top_tokens]
    rows = []
    for entry in top_tokens:
        rows.append(
            "<tr>"
            f"<td>{html.escape(entry['token'])}</td>"
            f"<td>{entry['activation']:.4f}</td>"
            f"<td>{entry['token_position']}</td>"
            f"<td>{html.escape(entry['text'])}</td>"
            "</tr>"
        )
    table = (
        "<table class='metrics-table'><thead><tr><th>token</th><th>activation</th><th>seq pos</th><th>text</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return svg_bar_chart(labels, values, "Top activating tokens", color="#16a34a") + table


def render_card_html(card: Dict) -> str:
    summary = card["summary"]
    histogram = card["activation_histogram"]
    histogram_svg = svg_bar_chart(histogram["bin_labels"], histogram["counts"], "Activation histogram", color="#7c3aed")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Latent {summary['latent_id']:05d}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px auto; max-width: 1100px; color: #111827; background: #f8fafc; }}
    a {{ color: #2563eb; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .metric {{ background: white; padding: 14px; border-radius: 10px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08); }}
    .metric .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }}
    .metric .value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    .section {{ background: white; padding: 18px; margin: 20px 0; border-radius: 12px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08); }}
    .token-seq {{ line-height: 2.2; margin-top: 8px; }}
    .token {{ padding: 4px 6px; border-radius: 6px; }}
    .metrics-table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    .metrics-table th, .metrics-table td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; vertical-align: top; }}
    .muted {{ color: #6b7280; }}
    .chart {{ width: 100%; height: auto; margin-top: 12px; }}
    .example-card {{ border-top: 1px solid #e5e7eb; padding-top: 16px; margin-top: 16px; }}
  </style>
</head>
<body>
  <p><a href="../index.html">Back to index</a></p>
  <h1>Latent {summary['latent_id']:05d}</h1>
  <p>Status: <strong>{html.escape(summary['status'])}</strong></p>
  <div class="summary">
    <div class="metric"><div class="label">Firing Rate</div><div class="value">{safe_float(summary['firing_rate']):.6f}</div></div>
    <div class="metric"><div class="label">Mean Act</div><div class="value">{safe_float(summary['mean_activation_when_active']):.4f}</div></div>
    <div class="metric"><div class="label">Max Act</div><div class="value">{safe_float(summary['max_activation']):.4f}</div></div>
    <div class="metric"><div class="label">Examples</div><div class="value">{safe_int(summary['example_count'])}</div></div>
  </div>
  <div class="section">
    <h2>Top Activating Examples</h2>
    {render_top_examples(card)}
  </div>
  <div class="section">
    <h2>Most Common High-Activation Tokens</h2>
    {render_top_tokens(card)}
  </div>
  <div class="section">
    <h2>Activation Histogram</h2>
    {histogram_svg}
  </div>
  <div class="section">
    <h2>Intervention Effects</h2>
    {render_intervention_section(card)}
  </div>
</body>
</html>"""


def render_index_html(summary_rows: Sequence[Dict], analysis_metadata: Dict, recon_metrics: Dict) -> str:
    rows = []
    for row in summary_rows:
        latent_id = safe_int(row["latent_id"])
        rows.append(
            "<tr>"
            f"<td><a href='cards/latent_{latent_id:05d}.html'>{latent_id:05d}</a></td>"
            f"<td>{html.escape(row['status'])}</td>"
            f"<td data-sort='{safe_float(row['firing_rate'])}'>{safe_float(row['firing_rate']):.6f}</td>"
            f"<td data-sort='{safe_float(row['mean_activation_when_active'])}'>{safe_float(row['mean_activation_when_active']):.4f}</td>"
            f"<td data-sort='{safe_float(row['max_activation'])}'>{safe_float(row['max_activation']):.4f}</td>"
            f"<td data-sort='{safe_int(row['example_count'])}'>{safe_int(row['example_count'])}</td>"
            f"<td data-sort='{safe_float(row['score'])}'>{safe_float(row['score']):.4f}</td>"
            "</tr>"
        )

    metadata_items = "".join(
        f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</li>"
        for key, value in analysis_metadata.items()
        if key not in {"requested_feature_ids", "intervention_feature_ids"}
    )
    recon_items = "".join(
        f"<li><strong>{html.escape(str(key))}:</strong> {html.escape(str(value))}</li>"
        for key, value in recon_metrics.items()
        if key != "validation_with_saev"
    )
    validation_json = html.escape(json.dumps(recon_metrics.get("validation_with_saev", {}), indent=2))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SAE Feature Cards</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px auto; max-width: 1280px; color: #111827; background: #f8fafc; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .panel {{ background: white; padding: 18px; border-radius: 12px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08); margin: 18px 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #e5e7eb; text-align: left; }}
    th button {{ background: none; border: none; font: inherit; cursor: pointer; color: #2563eb; padding: 0; }}
    code, pre {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
    pre {{ overflow: auto; padding: 12px; }}
  </style>
  <script>
    function sortTable(colIndex, numeric) {{
      const table = document.getElementById('latent-table');
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      const current = table.getAttribute('data-sort-dir') || 'desc';
      const next = current === 'asc' ? 'desc' : 'asc';
      rows.sort((a, b) => {{
        const aCell = a.cells[colIndex];
        const bCell = b.cells[colIndex];
        const aVal = aCell.dataset.sort || aCell.innerText;
        const bVal = bCell.dataset.sort || bCell.innerText;
        if (numeric) {{
          return next === 'asc' ? Number(aVal) - Number(bVal) : Number(bVal) - Number(aVal);
        }}
        return next === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }});
      rows.forEach(row => tbody.appendChild(row));
      table.setAttribute('data-sort-dir', next);
    }}
  </script>
</head>
<body>
  <h1>Layer-10 SAE Feature Cards</h1>
  <div class="panel">
    <h2>Analysis Metadata</h2>
    <ul>{metadata_items}</ul>
  </div>
  <div class="panel">
    <h2>Reconstruction Metrics</h2>
    <ul>{recon_items}</ul>
    <h3>Optional saev validation</h3>
    <pre>{validation_json}</pre>
  </div>
  <div class="panel">
    <h2>Latent Index</h2>
    <table id="latent-table" data-sort-dir="desc">
      <thead>
        <tr>
          <th><button onclick="sortTable(0, false)">latent</button></th>
          <th><button onclick="sortTable(1, false)">status</button></th>
          <th><button onclick="sortTable(2, true)">firing rate</button></th>
          <th><button onclick="sortTable(3, true)">mean act</button></th>
          <th><button onclick="sortTable(4, true)">max act</button></th>
          <th><button onclick="sortTable(5, true)">examples</button></th>
          <th><button onclick="sortTable(6, true)">score</button></th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>"""


def main():
    args = parse_args()
    analysis_dir = os.path.abspath(args.analysis_dir)
    out_dir = os.path.abspath(args.out_dir or os.path.join(analysis_dir, "html"))
    cards_out_dir = os.path.join(out_dir, "cards")
    os.makedirs(cards_out_dir, exist_ok=True)

    summary_rows = read_csv_rows(os.path.join(analysis_dir, "feature_summary.csv"))
    summary_rows.sort(key=lambda row: safe_float(row["score"]), reverse=True)
    analysis_metadata = read_json(os.path.join(analysis_dir, "analysis_metadata.json"))
    recon_metrics = read_json(os.path.join(analysis_dir, "recon_metrics.json"))

    card_paths = sorted(glob.glob(os.path.join(analysis_dir, "feature_cards", "latent_*.json")))
    for path in card_paths:
        card = read_json(path)
        latent_id = safe_int(card["latent_id"])
        card_html = render_card_html(card)
        with open(os.path.join(cards_out_dir, f"latent_{latent_id:05d}.html"), "w") as f:
            f.write(card_html)

    index_html = render_index_html(summary_rows, analysis_metadata, recon_metrics)
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(index_html)

    print(f"Rendered HTML feature cards to {out_dir}")


if __name__ == "__main__":
    main()
