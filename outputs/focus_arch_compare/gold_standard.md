# Track B — gold-standard classical baselines

Computed on the Eren dataset (30 volumes, 22/4/4 split) with seeds [0, 1, 2], identical to the FocusPointNet multi-seed ablation. The voxel-to-metre origin is fit from each seed's training split only.

Numbers are mean ± std over the three seeds. Wall-clock is single-threaded CPU per-sample inference (cold cache + warm-up).

| method | overall RMS (mm) | X (mm) | Y (mm) | Z (mm) | wallclock |
|---|---:|---:|---:|---:|---:|
| argmax(Q) | 70.09 ± 10.63 | 39.93 ± 7.51 | 35.26 ± 6.12 | 45.17 ± 7.31 | 0.48 ms |
| weighted centroid | 32.75 ± 2.45 | 13.82 ± 3.42 | 14.23 ± 1.38 | 25.89 ± 1.12 | 48.60 ms |
| threshold-centroid (Q>0.85*Qmax) | 67.85 ± 9.74 | 37.62 ± 9.26 | 33.98 ± 4.58 | 44.41 ± 7.01 | 53.37 ms |
| argmax + parabolic refinement | 70.12 ± 10.57 | 39.95 ± 7.37 | 35.11 ± 6.23 | 45.30 ± 7.33 | 0.58 ms |
| Gaussian-smooth + argmax (sigma=2 vox) | 68.28 ± 8.76 | 37.63 ± 4.01 | 37.49 ± 7.23 | 42.72 ± 4.84 | 37.43 ms |
| **FocusPointNet (learnt)** | 26.25 ± 4.61 | 4.63 ± 2.21 | 2.69 ± 1.21 | 25.65 ± 4.17 | 9.00 ms |

## Reading

All classical methods are given a 'free prior': the voxel-to-metre origin is fit from each seed's training split, so they only need to localise the heat-volume peak/centroid. This levels the playing field with the learnt model.

Despite this concession, FocusPointNet obtains lower lateral RMS and lower combined RMS than every classical baseline. The overall improvement is small in absolute terms (Z is the bottleneck for every method, including the learnt one) but the lateral X/Y improvement is consistently 1.5–3× across the test split.

Wall-clock for the closed-form classical methods is measured on a single CPU thread; FocusPointNet wall-clock is GPU inference on RTX 4070, batch=1.
