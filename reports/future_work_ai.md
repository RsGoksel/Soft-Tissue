# Future Work — Cutting-Edge Architectures and the Realistic Headroom for AI

This note addresses a recurring question that arose during the project:
*"Could we push the results further with a state-of-the-art model like
YOLO or some other frontier architecture, instead of the 3-D CNN and
U-Net we settled on?"*

The short answer is: **yes in some places, no in others, and the biggest
headroom is not where it is usually assumed to be**. This document
maps the landscape so that the next iteration of the project can invest
effort where the return is real.

---

## 1. Why YOLO is not the right tool for either of our tracks

YOLO is an object-detection family: it predicts axis-aligned bounding
boxes and class labels from an image. It is not an image-to-image
regressor and it is not a continuous-coordinate regressor. Our two
tracks are:

- **Track A** — 2-D forward pressure field prediction. This is a
  dense image-to-image regression (every pixel is a continuous Pa value).
  YOLO has no output structure for this.
- **Track B** — 3-D focus-point regression from the heat-deposition
  volume Q. The output is three continuous real numbers (x, y, z in
  metres). YOLO's bounding box is the wrong representation — we want
  a point, not an extent.

3-D medical YOLO variants exist (e.g. MedYOLO, 2024) for organ detection,
but they still output bounding boxes and are trained at much larger
data scales than we have.

## 2. Track A — architectures that could genuinely improve the 2-D FNO

At matched compute and data, the following have beaten FNO on published
PDE benchmarks:

| Model                       | Published gain over FNO | Relevance here |
|-----------------------------|-------------------------|----------------|
| **Transolver** (ICML 2024)  | 30–50 % lower rel-L2 on wave / Navier-Stokes tasks | High; physics-aware attention matches wave structure |
| **GNOT** (ICML 2023)        | 20–40 % lower rel-L2 on irregular geometries | High; geometry-aware |
| **UNO** (U-Net + FNO)       | 10–20 % lower rel-L2, robust across scales | Moderate; multi-scale spectral |
| **F-FNO** (factorised FNO)  | parity or slightly better, much lighter | Low — primarily a compute win |
| **PDE-Refiner** (diffusion) | strong on long rollouts | Low — we are not rolling out |

Translated to our numbers: the current **FNO test rel-L2 of 0.097**
could plausibly drop to **0.05–0.07** with Transolver or GNOT, at the
cost of 1–2 weeks of implementation and careful hyper-parameter work.
For a symposium paper this is worthwhile; for a clinical pipeline it
is a secondary concern, because the forward model is already far more
accurate than is needed to plan a focus.

**What is *not* expected to help**: generic image SOTA families
(ConvNeXt-V1, Swin-UNet, plain Transformers). We trained a ConvNeXt-2d
as an ablation and it reached only LpLoss 0.99 — an order of magnitude
worse than FNO. The spectral structure of the wave equation is what
FNO exploits, and generic CNN/ViT backbones have no such prior.

## 3. Track B — architectures and where the 30-sample regime breaks them

For the 3-D Q → focus-point problem we already tested five backbones at
matched ~0.9 M–1.5 M parameter budgets: plain CNN, ResNet-3D, a
multi-scale U-Net-encoder, a heatmap U-Net (DSNT loss), and a
heatmap+offset variant. Results (mean ± std where available):

| Model                     | Params | Test RMS (mm) | Z (mm) |
|---------------------------|--------|---------------|--------|
| Analytical (weighted centroid) | —    | 33.98        | 27.30  |
| **Plain 3-D CNN (multi-seed)** | 0.89 M | **26.25 ± 4.61** | 25.65 |
| ResNet-3D (multi-seed)    | 0.91 M | 34.48 ± 3.74  | 33.64  |
| Multi-scale UNet encoder  | 0.90 M | 34.17 ± 8.01  | 33.71  |
| Heatmap-DSNT U-Net        | 1.46 M | **25.27**     | **23.79** |
| Heatmap + offset (H3DE-style) | 1.46 M | 66.02     | 63.59  |

Two architecturally interesting backbones remain **untested**:

- **SwinUNETR** (MICCAI 2022 / NVIDIA MONAI). 3-D transformer built
  for medical imaging. At 30 samples, our expectation is 28–35 mm
  test RMS; it is parameter-heavy and attention is data-hungry.
- **U-Mamba** (state-space / selective-scan, 2024). Designed for long
  spatial dependencies in 3-D volumes. Expected 25–30 mm, i.e.
  comparable to our baseline. Worth an ablation run, not a pivot.

A naive 3-D Vision Transformer at 30 samples is expected to land at
40–60 mm — worse than baseline — because it has no inductive bias for
our problem.

## 4. Where the largest improvement actually lives: transfer learning

The bottleneck for Track B is **not architecture**, it is **sample
count**. Every backbone we tested clusters in the same 26–42 mm band
because 22 training samples simply cannot constrain a 0.9 M-parameter
3-D network's axial behaviour.

The single most promising lever is transfer from a pretrained 3-D
medical foundation model:

- **SAM-Med3D** (Meta / open source, 2024) — pretrained on ~100 k
  segmented medical volumes. Replace the segmentation head with a
  3-coordinate regression head; freeze the encoder, train the head,
  then fine-tune the top two encoder blocks.
- **NVIDIA MONAI / Auto3DSeg** — same idea, curated medical
  pretraining.

Published low-data fine-tunes of these models achieve **sub-3 mm
radial error on 20–50 sample regression tasks**. For our problem we
estimate a realistic drop of **test RMS from 26 mm to 10–15 mm** at
30 samples, and to **sub-5 mm** with the ~500 additional simulations
currently being generated.

**Why this dwarfs any architecture change at our data size:** the
current model learns Q-shape → focus-location from scratch on 22
samples. The foundation model already knows what medical 3-D volumes
look like; we only have to teach it *our specific regression head*.
The ablations across CNN / ResNet / UNet / heatmap already told us
the backbone family is not the issue.

## 5. Recommended priority for the next work cycle

Ranked by realistic return on effort:

1. **Transfer learning on Track B** (SAM-Med3D or MONAI encoder).
   Biggest expected gain, 2–3 days of engineering.
2. **More simulations** (push the Eren dataset from 30 to 500 samples).
   Enables every downstream improvement; already in progress.
3. **Transolver or GNOT on Track A**. Clear expected win on the paper
   metric, 1–2 weeks. Valuable for publication, modest clinical impact.
4. **Architecture exploration** (SwinUNETR, U-Mamba). Diminishing
   returns until items 1 and 2 are in place.

The guiding principle this project has settled on: **AI is not magic,
it is a mapping from inputs to outputs**. When the mapping is well
posed and the inductive bias matches (FNO for wave PDEs, or a
foundation model pretrained on matching data), AI can outperform
classical alternatives by large margins. When the problem is ill posed
(256-DOF phase regression with gauge symmetry) or the sample count is
too small for any prior-free model (30 volumes vs a 3-D CNN), no
amount of architectural sophistication rescues it.

This matches the empirical record of the project so far, and it is the
lens through which the next set of architecture choices should be made.
