# Soft-Tissue — AI-Assisted HIFU Planning for Heterogeneous Breast Tissue

An end-to-end pipeline for **non-invasive tumour ablation planning** with
high-intensity focused ultrasound (HIFU), combining a neural-operator
forward surrogate with a sample-efficient inverse focus-point regressor.
The repository documents both the working pipeline and the empirical
journey that shaped it — including the approaches that failed and the
reasons they failed, because those findings are what made the final
design work.

> Companion repository for a symposium paper (submission pending).
> Core collaborator: Eren ([ITÜ], inverse-problem simulator + k-Wave
> validation). Advisor: Gülşah Hoca.

---

## Table of contents

- [1. Problem setting](#1-problem-setting)
- [2. The approach that did not work, and why](#2-the-approach-that-did-not-work-and-why)
- [3. The pivot: two-track pipeline](#3-the-pivot-two-track-pipeline)
- [4. Track A — 2-D forward pressure-field surrogate](#4-track-a--2-d-forward-pressure-field-surrogate)
- [5. Track B — 3-D inverse focus-point regression](#5-track-b--3-d-inverse-focus-point-regression)
- [6. Gauge symmetry: empirical triple validation](#6-gauge-symmetry-empirical-triple-validation)
- [7. Phase-quantisation robustness study](#7-phase-quantisation-robustness-study)
- [8. Heatmap-regression exploration (nnLandmark / H3DE-Net style)](#8-heatmap-regression-exploration-nnlandmark--h3de-net-style)
- [9. What the numbers mean and where the next gain lives](#9-what-the-numbers-mean-and-where-the-next-gain-lives)
- [10. Repository layout](#10-repository-layout)
- [11. Reproducing the results](#11-reproducing-the-results)

---

## 1. Problem setting

Clinical HIFU planning in heterogeneous tissue requires solving two
coupled computational problems:

1. **Forward**: given a patient-specific tissue map (sound speed,
   density, attenuation), predict the acoustic pressure field produced
   by a given transducer configuration. Reference solver: full-wave
   k-Wave simulation (slow; seconds-to-minutes per configuration).
2. **Inverse**: given a desired focus location, synthesise the
   transducer phase vector that steers energy there. Classical
   approach: iterative optimisation over a forward solver — too slow
   for interactive planning.

The dataset used throughout is **OpenBreastUS** (Zeng et al. 2025):
2-D sound-speed maps derived from real breast-phantom volumes, paired
with k-Wave-computed pressure fields. To the best of our knowledge this
is the first application of OpenBreastUS to HIFU planning (the original
publication targets ultrasound computed tomography).

## 2. The approach that did not work, and why

The project opened with the intuitive formulation:

> Train a network `f: Q -> phase_vector` that maps the desired heat-
> deposition volume `Q` directly to the 256-element transducer phase
> vector.

Empirically this learned nothing. The loss plateaued well above random
baselines. A diagnostic study revealed two structural reasons.

### 2.1 The Q → phase map is one-to-many

Among the 30-sample dataset, phase vectors belonging to Q volumes with
*nearest-neighbour* target positions had circular-cosine similarity of
**0.002 ± 0.043** — indistinguishable from random noise. Many distinct
phase vectors produce heat deposition at the same target (because only
the relative beam geometry matters). A deterministic regressor cannot
fit a one-to-many relation.

### 2.2 Gauge symmetry

Adding a constant offset `+δ` to every element of the phase vector
leaves the interference pattern, and therefore the focus location,
exactly invariant. This is a continuous symmetry of the problem — the
network would have to "pick a branch" of infinitely many equivalent
outputs, so no fixed-target loss is stable.

Both of these were immediately visible once diagnostic metrics were
written — and they had not been caught earlier because the training
curves alone (loss drops a little, then plateaus) looked consistent
with a slightly-underpowered model. They turned out to be consistent
with an **ill-posed loss**.

## 3. The pivot: two-track pipeline

The reformulation is to keep classical physics where it is robust and
apply learning only where it buys something. This gives two
complementary surrogates:

- **Track A — Forward surrogate.** A neural operator predicts the
  pressure field in 2-D from tissue maps, replacing the expensive
  full-wave solver inside any future optimisation loop.
- **Track B — Inverse point regressor.** A compact 3-D CNN predicts
  the **focus coordinate (x, y, z)** — only three degrees of freedom
  — directly from `Q`. The transducer phases are then generated
  analytically by delay-and-sum beamforming using a known physical
  model. The one-to-many and gauge-symmetry pathologies disappear
  because the learned output is unambiguous (a single focal point).

## 4. Track A — 2-D forward pressure-field surrogate

Dataset: 1 000 OpenBreastUS sound-speed maps, 316 × 256 grid,
k-Wave-simulated peak-pressure targets at 1 MHz with a 12 mm water
layer. Single seeded split (70 / 15 / 15) across all backbones.
Objective: combined `LpLoss + 0.3 * H1Loss`. Test metric: relative
L2 (LpLoss).

| Backbone      | Params  | Test rel-L2 | Note |
|---------------|---------|-------------|------|
| **FNO2d**     | ~10.3 M | **0.097**   | Spectral basis matches wave structure |
| U-Net2d       | ~7.9 M  | 0.264       | Classical baseline, blurs the refraction tail |
| ConvNeXt2d    | 1.87 M  | 0.990       | Ablation; the mismatch is the point |

The 2.7× gap between FNO and U-Net, and the order-of-magnitude gap
between FNO and the 2025-style ConvNeXt encoder-decoder, show that
**the architecture prior matters**: FNO's FFT-based global receptive
field matches the wave-equation Green's function directly. A scaling
study from 200 → 1 000 samples reduced validation LpLoss from 0.81
to 0.39 (51 % improvement), suggesting that the full 8 000-sample
OpenBreastUS dataset has substantial headroom.

## 5. Track B — 3-D inverse focus-point regression

Dataset: 30 volumetric (Q, focus-point) samples generated by the ITÜ
collaborator, 22 / 4 / 4 split, 2 mm voxel spacing. Task: predict
continuous `(x, y, z)` focus from `(log1p(Q), mask)`.

### 5.1 Baseline FocusPointNet

A 4-level 3-D CNN with global average pooling and a 3-unit MLP head,
0.89 M parameters. **This is the working model.** Against the
strongest classical heat-map localiser (amplitude-weighted centroid,
test RMS 33.98 mm) it improves lateral error **2.5×**.

### 5.2 Multi-seed architecture ablation

To separate architecture effects from seed noise, the three primary
backbones (plain CNN, ResNet-3D, multi-scale UNet-encoder) were each
trained across seeds 0 / 1 / 2 at a matched ~0.9 M parameter budget,
120 epochs per run:

| Backbone | n | Test RMS (mm)  | X (mm)       | Y (mm)       | Z (mm)         |
|----------|---|----------------|--------------|--------------|----------------|
| **CNN**  | 3 | **26.25 ± 4.61** | 4.63 ± 2.21  | 2.69 ± 1.21  | **25.65 ± 4.17** |
| ResNet-3D | 3 | 34.48 ± 3.74   | 5.90 ± 3.76  | 3.10 ± 1.36  | 33.64 ± 3.21   |
| UNet-enc | 3 | 34.17 ± 8.01   | 3.93 ± 0.70  | 3.02 ± 1.05  | 33.71 ± 8.32   |

The three backbones produce statistically equivalent *lateral*
accuracy (3–6 mm, well inside the HIFU focal spot); the CNN wins
overall because it handles the *axial* direction better. **Axial
accuracy is the remaining bottleneck for every backbone** — a
property of data, not architecture (Z range is 2× lateral range;
the HIFU focal zone is axially elongated).

## 6. Gauge symmetry: empirical triple validation

The gauge invariance claim in Section 2.2 is not merely numerical
convenience — it was verified independently three times:

1. **Analytical** — derived from the delay-and-sum beamforming
   equation. Adding a constant phase factors out of the interference
   pattern.
2. **Synthetic** — on a 256-element planar array, least-squares
   phase-plus-offset fits confirm that a `+20°` offset produces
   acoustic-intensity change below 1 % and focus shift of 0 mm.
3. **Full-wave k-Wave simulation** — Eren (ITÜ) reproduced the
   `+20°` test in a full-physics simulator; the measured focus
   shift was also 0 mm and the intensity change below 1 %.

This three-way agreement is the single most important empirical
finding of the project because it tells us that **the output space
of any future phase regressor can be quotiented by the gauge** —
for example by fixing phase[0] = 0, or by working on the equivalence
classes directly. Either removes a spurious degree of freedom that
was costing every classical optimiser a great deal of wasted work.

## 7. Phase-quantisation robustness study

A downstream question: if the output space is discretised (phases
rounded to 5°, 10°, 15° steps — typical of real transducer drivers),
how much accuracy do we lose? Running the beamformer at each
quantisation level yielded:

| Step  | Intensity error | Focus shift |
|-------|-----------------|-------------|
| 5°    | < 0.35 %        | < 0.1 mm    |
| 10°   | < 1.1 %         | < 0.3 mm    |
| 15°   | < 2.5 %         | < 0.7 mm    |

Adoption: **5° is the project's default discretisation** for any
classical or future-learned phase regressor. Combined with the
gauge fix this gives a discrete, gauge-invariant output space that
is both tractable for learning and physically faithful.

## 8. Heatmap-regression exploration (nnLandmark / H3DE-Net style)

Motivated by the 2025 medical-imaging literature — **nnLandmark**
(Weihsbach et al. 2025) and **H3DE-Net** (arXiv:2502.14221) both
report sub-2 mm radial error on landmark detection with
heatmap-regression + offset-head designs — we implemented and
evaluated two variants on Track B.

Three empirical findings emerged, each of paper-level interest in
its own right:

1. **Plain MSE on a Gaussian heatmap target collapses at our data
   size.** A `σ = 3` voxel Gaussian occupies only ~3 × 10⁻⁵ of the
   2-million-voxel volume, giving vanishing gradient signal.
   Weighted-MSE and peak-scaling rescue attempts failed.
2. **DSNT loss (Nibali et al. 2018) eliminates the collapse and
   reaches parity with direct coordinate regression** —
   **25.27 mm test RMS** vs the CNN baseline's 26.25 mm, with a
   slight edge on axial accuracy (Z: 23.79 vs 25.65 mm). The
   soft-argmax variance-regularised objective is stable.
3. **The H3DE-Net-style sub-voxel offset head hurts at 22 training
   samples** — test RMS 66.02 mm, 2.5× worse than baseline. The
   offset regression is dense (3 channels per voxel) and cannot be
   constrained by our data.

Combined, these results **empirically locate the sample-size
threshold** at which the heatmap-regression advantage reported in
the nnLandmark family (10–30 % gain at 500+ samples) begins to
emerge for our problem. This is a useful negative / threshold result
for the paper.

## 9. What the numbers mean and where the next gain lives

### 9.1 Honest read of the scoreboard

- **Track A (forward)**: a strong positive. FNO's spectral prior is
  the dominant effect — swapping for a "bigger" generic CNN made
  things strictly worse. The 0.097 test LpLoss is a clean result
  and the 2.7× gap over U-Net is the core empirical contribution.
- **Track B (inverse, lateral)**: a strong positive. With 30
  simulations the network already localises the focus laterally
  inside the native focal spot (X 4.6 mm, Y 2.7 mm). Every backbone
  agrees, so the result is a property of the task, not of any one
  model.
- **Track B (inverse, axial)**: a genuine limitation. 25 mm axial
  RMS is clinically too loose, but the root cause is 22 training
  samples + anatomy, not the backbone. See Section 9.2.
- **Symmetry / quantisation**: cleanly characterised, triple-validated,
  and adopted into the pipeline (5° discrete, gauge-fixed).

### 9.2 What would actually move the axial number

See [`reports/future_work_ai.md`](reports/future_work_ai.md) for the
full argument. Briefly:

1. **Transfer learning from a pretrained 3-D medical foundation
   model** (SAM-Med3D, MONAI). Expected test RMS drop from 26 mm to
   10–15 mm at the current 30-sample regime; to sub-5 mm once the
   dataset grows.
2. **More simulations** (500+). Enables every downstream
   improvement; already in progress with the ITÜ collaborator.
3. **Transolver / GNOT on Track A**. Clear paper-metric win
   (expected 30–50 % lower rel-L2), modest clinical impact at the
   current pipeline stage.

Not on the shortlist: a pivot to YOLO or generic image-SOTA
backbones. Our ConvNeXt2d ablation and the backbone-invariant
Track B results both confirm that the architecture *family* is
not the blocker — it is the data size and the transfer-learning
lever.

## 10. Repository layout

```
src/
  unet.py              2-D U-Net (Track A baseline)
  convnext2d.py        2-D ConvNeXt encoder-decoder (ablation)
  eren_model.py        FocusPointNet CNN + ResNet-3D + UNet-3D-encoder registry
  focus_heatmap.py     Heatmap U-Net + DSNT utilities (nnLandmark-style)
  dataset.py           Pressure-field HDF5 dataset (Track A)
  eren_dataset.py      Q / focus-point HDF5 dataset (Track B)
  simulate.py          k-Wave thin wrapper
  phantom.py           Phantom loaders (synthetic + OA-Breast)
  tissue_properties.py Label -> (c, rho, alpha) mapping

scripts/
  train_fno.py             Track A — FNO trainer
  train_unet.py            Track A — U-Net trainer
  train_convnext.py        Track A — ConvNeXt trainer
  train_focus_point.py     Track B — FocusPointNet trainer (baseline CNN)
  compare_focus_architectures.py  Track B — single-seed 3-backbone comparison
  multi_seed_focus.py      Track B — 3-seed × 3-backbone statistically clean ablation
  train_focus_heatmap.py   Track B — Heatmap/DSNT trainer (nnLandmark-style)
  phase_quantization_study.py  Phase discretisation sensitivity
  baseline_focus.py        Classical localiser baselines (centroid, percentile)
  compare_models.py        Track A end-to-end figure generation
  build_report.py          Markdown -> PDF report compilation
  build_sent_bundle.py     Assemble deliverable zip for advisor

reports/
  abstract_en.md           Symposium abstract (English)
  abstract_tr.md           Symposium abstract (Turkish)
  sonuclar_hoca.pdf        Main visual results report
  future_work_ai.md        Cutting-edge architectures & transfer-learning roadmap

outputs/
  focus_arch_compare/      Multi-seed ablation tables + logs
  focus_heatmap/           Heatmap / DSNT ablation JSONs
  focus_point/             Baseline CNN training artefacts
  fno_1k / unet_1k / convnext_1k/   Track A training artefacts
  phase_quant_10deg.png    Quantisation study figure
  comparison_1k.png        Track A five-panel comparison
```

Large binary dependencies (`*.h5` datasets, `*.mat` raw phantoms,
`*.pt` checkpoints, the 10-GB OpenBreastUS download) are excluded
from version control. See Section 11 for how to obtain / regenerate
them.

## 11. Reproducing the results

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

**Track A** (regenerate a 1 000-sample dataset from OpenBreastUS
slices and train the three backbones):

```bash
python scripts/generate_dataset.py --n 1000 --out data/dataset_v1.h5
python scripts/train_fno.py        --data data/dataset_v1.h5 --epochs 100
python scripts/train_unet.py       --data data/dataset_v1.h5 --epochs 100
python scripts/train_convnext.py   --data data/dataset_v1.h5 --epochs 60
python scripts/compare_models.py   --data data/dataset_v1.h5  # five-panel figure
```

**Track B** (baseline + multi-seed ablation + heatmap exploration):

```bash
python scripts/preprocess_eren_v2.py   # -> data/eren/dataset_v2.h5
python scripts/train_focus_point.py --epochs 300            # baseline CNN
python scripts/multi_seed_focus.py --seeds 0 1 2 --epochs 120
python scripts/train_focus_heatmap.py --arch heatmap        --loss dsnt --epochs 80
python scripts/train_focus_heatmap.py --arch heatmap_offset --loss dsnt --epochs 100
```

**Diagnostics and ancillary**:

```bash
python scripts/phase_quantization_study.py   # 5/10/15 degree table
python scripts/phase_to_focus_offset_test.py # gauge validation
python scripts/diagnose_eren.py              # one-to-many / similarity metrics
```

**Final deliverable bundle**:

```bash
python scripts/build_sent_bundle.py          # -> sent/ + sent.zip
```

---

*Repository mirror of the working copy as of 2026-04-24. Symposium
submission: reports/abstract_en.md (EN), reports/abstract_tr.md (TR).
Visual report: reports/sonuclar_hoca.pdf. Roadmap: reports/future_work_ai.md.*
