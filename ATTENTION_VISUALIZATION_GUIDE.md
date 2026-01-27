# TaxaAdapter Cross-Attention Visualization Guide

This document explains the attention visualization outputs and how to interpret them for your CVPR rebuttal.

## Background: What Are We Visualizing?

In TaxaAdapter, the **IP-Adapter mechanism** injects taxonomic information (from BioCLIP) into the image generation process through **cross-attention**.

The cross-attention works as follows:
- **Query (Q)**: Spatial features from the image being generated (each position in the latent space)
- **Key (K)**: Projected taxonomy embeddings (4 tokens)
- **Value (V)**: Projected taxonomy embeddings (4 tokens)

The attention map shows: **"Which parts of the generated image attend to which taxonomy tokens?"**

```
Attention = softmax(Q @ K^T / sqrt(d))
```

---

## Script 1: `inference_attention_viz.py`

### Output Files

For each species, you get a folder with:

#### 1. `generated.png`
The generated image using the full taxonomic name.

#### 2. `attention_all_tokens.png`
A grid visualization showing:

```
+------------------+----------+----------+----------+----------+
| Generated Image  | Token 1  | Token 2  | Token 3  | Token 4  |
+------------------+----------+----------+----------+----------+
| Taxonomy Info    | Attn Map | Attn Map | Attn Map | Attn Map |
|  Kingdom: ...    |    1     |    2     |    3     |    4     |
|  Phylum: ...     |          |          |          |          |
|  ...             |          |          |          |          |
+------------------+----------+----------+----------+----------+
```

**How to interpret:**
- **Top row**: Heatmap overlays showing WHERE in the image each token focuses
- **Bottom row**: Raw attention maps (red = high attention, blue = low)
- **Token 1-4**: The 4 projected tokens from the taxonomy embedding

**What to look for:**
- Different tokens should focus on DIFFERENT parts of the image
- Token 1-2 often capture broader/structural features
- Token 3-4 often capture finer/detailed features
- If all tokens look the same = taxonomy isn't well-disentangled

#### 3. `attention_per_level.png`
Similar to above but with interpretive labels suggesting what each token might encode.

#### 4. `attention_token_N_overlay.png` (N=1,2,3,4)
Individual heatmap overlays for each token.
- **Red/Yellow areas**: High attention (the model "looks at" these regions when using this token)
- **Blue areas**: Low attention

#### 5. `attention_token_N_raw.png` (N=1,2,3,4)
Raw attention maps without image overlay. Useful for quantitative analysis.

---

## Script 2: `inference_taxonomic_level_analysis.py`

This script is MORE IMPORTANT for the reviewer's question. It shows how attention changes as we progressively add more taxonomic information.

### Output Files

#### 1. `level_N_LevelName.png` (N=1-7)
Generated images at each taxonomic level:
- Level 1: Just "Animalia" (Kingdom)
- Level 2: "Animalia Chordata" (Kingdom + Phylum)
- Level 3: "Animalia Chordata Aves" (+ Class)
- ...
- Level 7: Full species name

**What to look for:**
- Images should become MORE SPECIFIC as levels increase
- Level 1-3: Generic animal/bird shape
- Level 4-5: More specific body type
- Level 6-7: Species-specific features (colors, patterns)

#### 2. `level_comparison.png`
The MAIN visualization for your rebuttal. A 3-row grid:

```
Row 1: Generated images at each level (Kingdom → Species)
Row 2: Attention maps at each level
Row 3: DIFFERENCE maps showing what each level ADDS
```

**Row 1 - Generated Images:**
Shows how the image evolves from generic to specific.

**Row 2 - Attention Maps:**
Shows WHERE the model focuses at each taxonomic level.
- Early levels (Kingdom, Phylum): Diffuse attention over whole image
- Later levels (Genus, Species): Focused attention on discriminative features

**Row 3 - Difference Maps (MOST IMPORTANT!):**
Shows what CHANGES when we add each new taxonomic level.
- **Red**: Areas that get MORE attention with this level
- **Blue**: Areas that get LESS attention with this level

Example interpretation:
- "+Class (Aves)": Red on wings/beak = "adding 'bird' info makes model focus on bird-specific parts"
- "+Species": Red on specific color patterns = "species info focuses on unique markings"

#### 3. `attention_statistics.png`
Quantitative plots showing how attention properties change across levels:

**Plot 1: Mean Attention**
- Average attention intensity across the image
- Should stay relatively stable

**Plot 2: Attention Std Dev (Focus Sharpness)**
- Higher = attention is more concentrated/focused
- **Expected pattern**: INCREASES from Kingdom → Species
- This shows: "More specific taxonomy = more focused attention"

**Plot 3: Max Attention (Peak Intensity)**
- Maximum attention value
- Higher peaks = stronger focus on specific regions

**Plot 4: Attention Entropy (Spread)**
- Higher entropy = attention spread across whole image
- Lower entropy = attention concentrated on few areas
- **Expected pattern**: DECREASES from Kingdom → Species
- This shows: "Species-level info creates localized, discriminative attention"

#### 4. `attention_overlays.png`
Heatmap overlays for each taxonomic level on the final generated image.

---

## How to Use This for Your Rebuttal

### Addressing the Reviewer's Question

> "Whether specific taxonomy tokens encoded specific visual traits by visualizing cross-attention maps"

**Answer with these visualizations:**

1. **Show `level_comparison.png`** - The difference maps (Row 3) directly show that:
   - Adding Kingdom/Phylum creates BROAD attention
   - Adding Family/Genus creates attention on BODY STRUCTURE
   - Adding Species creates attention on FINE-GRAINED features (patterns, colors)

2. **Show `attention_statistics.png`** - The quantitative plots demonstrate:
   - Attention becomes MORE FOCUSED (higher std, lower entropy) with finer taxonomy
   - This proves taxonomy tokens encode hierarchical visual information

3. **Show `attention_all_tokens.png`** - Individual tokens capture different aspects:
   - Some tokens focus on overall shape
   - Other tokens focus on texture/patterns
   - This shows the projection model learns to disentangle features

### Example Rebuttal Text

> "Following the approach of ConceptAttention [2] and PartCraft [3], we visualize cross-attention maps between taxonomy tokens and generated images. Figure X shows that:
>
> (1) Attention becomes progressively more focused as taxonomic specificity increases (entropy decreases from Kingdom to Species level, see Fig X-b).
>
> (2) Difference maps reveal that higher taxonomic levels (Kingdom, Phylum) activate broad image regions, while lower levels (Genus, Species) activate species-discriminative features such as color patterns and body markings (Fig X-a, Row 3).
>
> (3) The 4 projected tokens learn complementary representations, with different tokens attending to structural vs. textural features (Fig X-c).
>
> These visualizations confirm that TaxaAdapter successfully encodes hierarchical taxonomic information into visually meaningful attention patterns."

---

## Interpreting Color Maps

### Attention Heatmaps (Jet colormap)
```
Blue (0.0) → Cyan → Green → Yellow → Red (1.0)
   Low                              High
Attention                        Attention
```

### Difference Maps (RdBu_r colormap)
```
Blue (-0.5) → White (0.0) → Red (+0.5)
 Decreased      No change    Increased
 attention                   attention
```

---

## Troubleshooting

**Q: All attention maps look similar?**
- The model may not have learned good disentanglement
- Try different species with more visual variation
- Check if the IP-Adapter was trained long enough

**Q: Attention is uniform everywhere?**
- Scale parameter might be too low (try `--scale 1.0`)
- The taxonomy might be too generic

**Q: No attention captured?**
- Check SLURM output for errors
- Ensure `--num_tokens` matches training (default: 4)

---

## References

[2] ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features. ICML 2025.
[3] PartCraft: Crafting Creative Objects by Parts. ECCV 2024.
