"""
Train a FocusPointNet to regress a HIFU target point (x, y, z) from the
heat-deposition volume Q.

Why this task and not phase prediction? See the 2026-04-17 diagnostic
(scripts/diagnose_eren.py):
    - H1 Q -> target_pt: RMS 9.9 mm on 30 samples (LEARNABLE)
    - H2 target_pt -> phase vector: nearest-bin cosine sim 0.003 (~random)
So phase prediction is one-to-many and not a function, but focus prediction
is well-posed and sample-efficient. A beamformer can then analytically turn
the predicted 3-D focus into 256 transducer phases.

Usage:
    python scripts/train_focus_point.py \
        --data data/eren/dataset_v2.h5 --epochs 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eren_dataset import ErenPhaseDataset                   # noqa: E402
from eren_model import FocusPointNet                        # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/eren/dataset_v2.h5"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/focus_point"))
    return p.parse_args()


def split_indices(n: int, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    n_val  = max(1, int(round(val_frac  * n)))
    return (
        idx[: n - n_val - n_test],
        idx[n - n_val - n_test : n - n_test],
        idx[n - n_test :],
    )


def denorm_rms_mm(pred_std: torch.Tensor,
                  true_std: torch.Tensor,
                  tgt_std: np.ndarray) -> float:
    """Un-standardise both sides, then report RMS position error in mm."""
    std_t = torch.as_tensor(tgt_std, dtype=pred_std.dtype,
                            device=pred_std.device)
    err_m = (pred_std - true_std) * std_t        # back to metres
    rms_m = torch.sqrt((err_m ** 2).sum(dim=1).mean())
    return float(rms_m * 1000.0)                 # mm


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[focus] device={device}")

    import h5py
    with h5py.File(args.data, "r") as f:
        n = f["Q"].shape[0]
    tr, va, te = split_indices(n, args.val_frac, args.test_frac, args.seed)
    print(f"[focus] n={n}  train={tr.size}  val={va.size}  test={te.size}")

    train_ds = ErenPhaseDataset(args.data, indices=tr)
    val_ds   = ErenPhaseDataset(args.data, indices=va, stats=train_ds.stats)
    test_ds  = ErenPhaseDataset(args.data, indices=te, stats=train_ds.stats)

    if train_ds._t_cache is None:
        raise SystemExit("target_pt_m missing from dataset -- regenerate HDF5")

    tgt_std_np = np.asarray(train_ds.stats["tgt_std"], dtype=np.float32)
    tgt_mean_np = np.asarray(train_ds.stats["tgt_mean"], dtype=np.float32)
    print(f"[focus] tgt_mean (m) = {tgt_mean_np}")
    print(f"[focus] tgt_std  (m) = {tgt_std_np}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    model = FocusPointNet(in_channels=2,
                          base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[focus] FocusPointNet params: {n_params / 1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val_loss": [], "val_rms_mm": []}
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["target_pt"].to(device)
            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            # gradient clipping -- small batch + BN occasionally spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        va_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["target_pt"].to(device)
                pred = model(x)
                va_loss += loss_fn(pred, y).item()
                preds.append(pred); trues.append(y)
        va_loss /= max(1, len(val_loader))
        va_pred = torch.cat(preds, dim=0)
        va_true = torch.cat(trues, dim=0)
        va_rms_mm = denorm_rms_mm(va_pred, va_true, tgt_std_np)

        sched.step()
        history["train"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_rms_mm"].append(va_rms_mm)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  "
              f"train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"val RMS {va_rms_mm:5.2f} mm")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "stats": train_ds.stats,
                    "args": vars(args),
                },
                args.out_dir / "best.pt",
            )

    # -- test
    ckpt = torch.load(args.out_dir / "best.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    te_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["target_pt"].to(device)
            pred = model(x)
            te_loss += loss_fn(pred, y).item()
            preds.append(pred); trues.append(y)
    te_loss /= max(1, len(test_loader))
    te_pred = torch.cat(preds, dim=0)
    te_true = torch.cat(trues, dim=0)
    te_rms_mm = denorm_rms_mm(te_pred, te_true, tgt_std_np)
    # per-axis RMS (mm)
    per_axis_err_m = (te_pred - te_true).cpu().numpy() * tgt_std_np
    per_axis_rms_mm = np.sqrt((per_axis_err_m ** 2).mean(axis=0)) * 1000.0
    print(f"[focus] TEST MSE {te_loss:.4f}  RMS {te_rms_mm:5.2f} mm  "
          f"per-axis RMS (x,y,z) = {per_axis_rms_mm.round(2)}")

    # --- loss curve ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MSE (standardised)"); ax[0].legend()
    ax[0].set_title("Training loss")
    ax[1].plot(history["val_rms_mm"], color="tab:red")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val RMS [mm]")
    ax[1].set_title("Val focus-point RMS error")
    ax[1].axhline(10, color="grey", linestyle="--", alpha=0.6,
                  label="10 mm = ~1 voxel @ 0.5 mm")
    fig.tight_layout()
    fig.savefig(args.out_dir / "loss_curve.png", dpi=120)

    # --- scatter: pred vs true in physical mm ---
    te_pred_m = te_pred.cpu().numpy() * tgt_std_np + tgt_mean_np
    te_true_m = te_true.cpu().numpy() * tgt_std_np + tgt_mean_np
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    names = ("X", "Y", "Z")
    for i, a in enumerate(ax):
        a.scatter(te_true_m[:, i] * 1000, te_pred_m[:, i] * 1000,
                  s=40, alpha=0.8)
        lo = min(te_true_m[:, i].min(), te_pred_m[:, i].min()) * 1000
        hi = max(te_true_m[:, i].max(), te_pred_m[:, i].max()) * 1000
        a.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
        a.set_xlabel(f"target {names[i]} [mm]")
        a.set_ylabel(f"predicted {names[i]} [mm]")
        a.set_title(f"{names[i]}  RMS={per_axis_rms_mm[i]:.2f} mm")
    fig.tight_layout()
    fig.savefig(args.out_dir / "test_scatter.png", dpi=120)

    print(f"[focus] saved checkpoint + figures to {args.out_dir}")


if __name__ == "__main__":
    main()
