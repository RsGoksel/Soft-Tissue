"""
Go/no-go diagnostic for the HIFU inverse problem.

Tests two hypotheses in order:
  H1. Q -> target_pt (x, y, z) is learnable. If a 3D CNN cannot recover the
      3 focus coordinates from Q with 22 training samples, the problem is
      either very noisy or the Q volumes are nearly identical - in either
      case predicting 256 phases is hopeless.
  H2. Nearby target points have correlated phase vectors. We compute
      |corr(phase_i, phase_j)| as a function of ||target_i - target_j|| using
      the CSV alone (no .mat needed). If phases look uncorrelated even for
      nearest neighbours, the Q->phase map is not a function - it is
      one-to-many and cannot be regressed.

Usage:
    python scripts/diagnose_eren.py \
        --data data/eren/dataset_v2.h5 \
        --csv hifu_phase_dataset_1mm.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from eren_dataset import ErenPhaseDataset  # noqa: E402
from eren_model import Conv3dBlock  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/eren/dataset_v2.h5"))
    p.add_argument("--csv", type=Path,
                   default=Path("hifu_phase_dataset_1mm.csv"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    return p.parse_args()


# ---------- H1: train a tiny Q->xyz regressor -------------------------- #
class TargetPtRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            Conv3dBlock(2, 16), nn.MaxPool3d(2),
            Conv3dBlock(16, 32), nn.MaxPool3d(2),
            Conv3dBlock(32, 64), nn.MaxPool3d(2),
            Conv3dBlock(64, 128), nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


def h1_test(args) -> None:
    print("\n=== H1: Q -> target_pt learnability test ===")
    with h5py.File(args.data, "r") as f:
        n = f["Q"].shape[0]
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    n_val = max(2, n // 5)
    tr, va = idx[:-n_val], idx[-n_val:]
    print(f"n={n}  train={tr.size}  val={va.size}")

    ds_tr = ErenPhaseDataset(args.data, indices=tr)
    ds_va = ErenPhaseDataset(args.data, indices=va, stats=ds_tr.stats)

    if ds_tr._t_cache is None:
        print("  target_pt_m missing from h5 - cannot run H1.")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TargetPtRegressor().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    dl_tr = DataLoader(ds_tr, batch_size=4, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=4)

    best_va = float("inf")
    for ep in range(args.epochs):
        model.train()
        tl = 0.0
        for b in dl_tr:
            x = b["x"].to(dev)
            t = b["target_pt"].to(dev)  # already standardised
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(x), t)
            loss.backward()
            opt.step()
            tl += loss.item()
        tl /= len(dl_tr)

        model.eval()
        with torch.no_grad():
            vl = 0.0
            for b in dl_va:
                x = b["x"].to(dev)
                t = b["target_pt"].to(dev)
                vl += nn.functional.mse_loss(model(x), t).item()
            vl /= max(1, len(dl_va))
        if vl < best_va:
            best_va = vl
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1:3d}  train {tl:.4f}  val {vl:.4f}")

    # un-standardise val error to metres
    std = ds_tr.stats["tgt_std"]
    rms_m = float(np.sqrt(best_va) * np.linalg.norm(std))
    print(f"  best val MSE (standardised) = {best_va:.4f}")
    print(f"  approx RMS target-pt error  = {rms_m*1000:.1f} mm")
    print("  GO signal: RMS < 10 mm.  NO-GO: RMS >= 30 mm (problem is "
          "not learnable at this sample count).")


# ---------- H2: phase correlation vs target distance ------------------ #
def h2_test(args) -> None:
    print("\n=== H2: phase-vector correlation vs target-pt distance ===")
    df = pd.read_csv(args.csv)
    df = df[(df["Algorithm"] == 1) & (df["NoiseType"] == 0)].reset_index(drop=True)
    print(f"  clean subset rows: {len(df)}")
    if len(df) < 20:
        print("  too few rows, skipping H2")
        return

    tgt = df[["TargetX_m", "TargetY_m", "TargetZ_m"]].to_numpy() \
        if "TargetX_m" in df.columns else None
    if tgt is None:
        # fallback: parse TargetXYZ_m if stored as string tuple
        col = df["TargetXYZ_m"].astype(str).str.strip("()[] ").str.split(r"[, ]+")
        tgt = np.array([[float(v) for v in r if v] for r in col])
    phase_cols = [f"Phase_{i}" for i in range(1, 257)]
    P = df[phase_cols].to_numpy()           # (N, 256)

    # circular phasor per transducer, then cosine-similarity between rows
    z = np.exp(1j * P)                      # (N, 256)
    z /= np.abs(z).clip(1e-9)
    # row-wise mean-phase removal (fix global gauge)
    mean_phase = np.angle(z.mean(axis=1, keepdims=True))
    z_dg = z * np.exp(-1j * mean_phase)

    # pairwise cosine similarity (real part of inner product / 256)
    sim = (z_dg @ z_dg.conj().T).real / 256.0

    # pairwise target distances
    d = np.linalg.norm(tgt[:, None] - tgt[None, :], axis=-1)

    iu = np.triu_indices(len(df), k=1)
    sim_flat = sim[iu]
    d_flat = d[iu]

    # bin by distance
    edges = np.quantile(d_flat, np.linspace(0, 1, 6))
    print("  distance_bin_m   mean_sim   std_sim   n")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_flat >= lo) & (d_flat < hi)
        if m.sum() == 0:
            continue
        print(f"  [{lo:.4f},{hi:.4f}]   "
              f"{sim_flat[m].mean():+.3f}    {sim_flat[m].std():.3f}   "
              f"{m.sum()}")
    print("  Interpret: if mean_sim ~ 0 in the NEAREST bin, phases are "
          "effectively random given target_pt - Q->phase is NOT a function.")


def main() -> None:
    args = parse_args()
    h1_test(args)
    h2_test(args)


if __name__ == "__main__":
    main()
