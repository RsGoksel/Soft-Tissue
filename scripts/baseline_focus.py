"""
Analytical baselines for the Q -> focus_point task.

Baselines (no learning, no training data):
    1. argmax(Q)             -- single voxel with the max heat value
    2. weighted centroid     -- centre of mass of Q
    3. threshold-centroid    -- centroid of voxels with Q > c * Qmax

Each baseline produces voxel coordinates. To convert to metres we need
spacing and origin. Spacing is known (2 mm = 1 mm native * downsample 2).
Origin is not stored in the h5, so we FIT IT FROM TRAINING DATA ONLY:

    origin = mean_train( target_pt_m - voxel_pred * spacing )

This is fair for a baseline: it lets the baseline "know the crop geometry"
which is a free prior, then its only remaining job is to localise the
peak/centroid of Q inside the volume.

We also note that in the stored h5 Q is log1p-compressed (monotonic), so
argmax / centroid of log Q == argmax / centroid of Q (for centroid the
weights differ but the location still tracks the peak closely).

Outputs printed:
    per-axis RMS for each baseline on the test set + FocusPointNet row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


DATA = PROJECT_ROOT / "data" / "eren" / "dataset_v2.h5"
VOXEL_MM = 2.0           # 1 mm native * downsample 2
SPACING = np.array([VOXEL_MM, VOXEL_MM, VOXEL_MM]) * 1e-3  # metres, (z, y, x)

# Axis ordering we assume for mapping (voxel axis 0, 1, 2) -> (z, y, x).
# Target vector convention in the h5 is target_pt_m = (x, y, z).
# So voxel (i0, i1, i2) maps to target_axes permuted accordingly.
VOXEL_AXIS_TO_TGT = (2, 1, 0)  # voxel (axis0=z, axis1=y, axis2=x) -> target (x, y, z)


def split_indices(n: int, val_frac=0.15, test_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    n_val  = max(1, int(round(val_frac  * n)))
    return (
        idx[: n - n_val - n_test],
        idx[n - n_val - n_test : n - n_test],
        idx[n - n_test :],
    )


def rms_mm(pred_m: np.ndarray, true_m: np.ndarray) -> tuple[float, np.ndarray]:
    err = (pred_m - true_m) * 1000.0
    per_axis = np.sqrt((err ** 2).mean(axis=0))
    overall  = float(np.sqrt((err ** 2).sum(axis=1).mean()))
    return overall, per_axis


def voxel_pred_to_metres(voxels: np.ndarray) -> np.ndarray:
    """voxels (N, 3) with axis order (i0=z, i1=y, i2=x) -> metres (x, y, z)."""
    out = np.zeros_like(voxels, dtype=np.float32)
    for voxel_axis, tgt_axis in enumerate(VOXEL_AXIS_TO_TGT):
        out[:, tgt_axis] = voxels[:, voxel_axis] * SPACING[voxel_axis]
    return out


def argmax_voxel(Q: np.ndarray) -> np.ndarray:
    N = Q.shape[0]
    out = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        k = int(Q[i].argmax())
        out[i] = np.unravel_index(k, Q[i].shape)
    return out


def centroid_voxel(Q: np.ndarray, threshold: float | None = None) -> np.ndarray:
    N = Q.shape[0]
    out = np.zeros((N, 3), dtype=np.float32)
    zz = yy = xx = None
    for i in range(N):
        v = Q[i].astype(np.float64)
        # Q is stored as log1p(Q/1e3). Undo to get back to (near-)linear heat
        # before taking a weighted centroid -- otherwise the log compression
        # over-weights the tail.
        v = np.expm1(v)
        if threshold is not None:
            thr = threshold * v.max()
            v = np.where(v >= thr, v, 0.0)
        total = v.sum()
        if total <= 0:
            continue
        if zz is None:
            zz, yy, xx = np.meshgrid(
                np.arange(v.shape[0]),
                np.arange(v.shape[1]),
                np.arange(v.shape[2]),
                indexing="ij",
            )
        out[i, 0] = (zz * v).sum() / total
        out[i, 1] = (yy * v).sum() / total
        out[i, 2] = (xx * v).sum() / total
    return out


def fit_origin(train_voxels: np.ndarray, train_tgt_m: np.ndarray) -> np.ndarray:
    """origin = mean( target - voxel_pred_m ) over train."""
    pred_m = voxel_pred_to_metres(train_voxels)
    return (train_tgt_m - pred_m).mean(axis=0).astype(np.float32)


def evaluate(name: str, Q: np.ndarray, tgt_m: np.ndarray,
             tr_idx: np.ndarray, te_idx: np.ndarray,
             voxel_fn) -> None:
    vox_all = voxel_fn(Q)
    origin = fit_origin(vox_all[tr_idx], tgt_m[tr_idx])
    pred_m = voxel_pred_to_metres(vox_all) + origin
    overall, per_axis = rms_mm(pred_m[te_idx], tgt_m[te_idx])
    print(f"{name:<30} {overall:>8.2f}   "
          f"({per_axis[0]:5.2f}, {per_axis[1]:5.2f}, {per_axis[2]:5.2f})   "
          f"origin fit = {origin.round(4).tolist()}")


def main() -> None:
    with h5py.File(DATA, "r") as f:
        Q = np.asarray(f["Q"], dtype=np.float32)
        tgt_m = np.asarray(f["target_pt_m"], dtype=np.float32)

    n = Q.shape[0]
    tr, _va, te = split_indices(n)
    print(f"[baseline] total={n}  train={tr.size}  test={te.size}")
    print(f"[baseline] Q shape={Q.shape}  spacing={SPACING.tolist()} m\n")

    print(f"{'baseline':<30} {'RMS tot':>8}   per-axis (X, Y, Z) [mm]")
    print("-" * 90)

    evaluate("argmax(Q)",                  Q, tgt_m, tr, te, argmax_voxel)
    evaluate("weighted centroid",          Q, tgt_m, tr, te,
             lambda q: centroid_voxel(q))
    evaluate("centroid (Q > 0.5 Qmax)",    Q, tgt_m, tr, te,
             lambda q: centroid_voxel(q, 0.5))
    evaluate("centroid (Q > 0.8 Qmax)",    Q, tgt_m, tr, te,
             lambda q: centroid_voxel(q, 0.8))

    print("-" * 90)
    print(f"{'FocusPointNet (learnt)':<30} {27.89:>8.2f}   "
          f"({5.09:5.2f}, {5.28:5.2f}, {26.91:5.2f})   [for reference]")
    print("\nNote: analytic baselines know the crop origin (fit from "
          "training data). They only have to find the Q peak / centroid.")


if __name__ == "__main__":
    main()
