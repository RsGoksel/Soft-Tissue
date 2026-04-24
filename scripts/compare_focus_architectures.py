"""
Apples-to-apples benchmark of the three FocusPointNet variants:
  * cnn     -- baseline 4-level 3D CNN (reference)
  * resnet  -- ResNet3D encoder (identity skips)
  * unet    -- multi-scale UNet-style encoder with concat-pool head

All three use base_channels=16 (~0.9M params) and share:
  - data split (seed=0, 70/15/15)
  - optimiser (AdamW 3e-4, weight_decay 1e-4)
  - cosine LR schedule
  - gradient clipping (max_norm=1.0)
  - same epochs, same batch size

Why: so any per-axis RMS difference between the three is attributable to
architecture only. This is the "backbone robustness" check the advisor asked for.

Usage:
    python scripts/compare_focus_architectures.py --epochs 150
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eren_dataset import ErenPhaseDataset             # noqa: E402
from eren_model import FOCUS_ARCH_REGISTRY            # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=PROJECT_ROOT / "data" / "eren" / "dataset_v2.h5")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "outputs" / "focus_arch_compare")
    return p.parse_args()


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


def rms_mm_per_axis(pred_std: torch.Tensor, true_std: torch.Tensor,
                    tgt_std: np.ndarray) -> tuple[float, np.ndarray]:
    std_t = torch.as_tensor(tgt_std, dtype=pred_std.dtype,
                            device=pred_std.device)
    err_m = (pred_std - true_std) * std_t
    err_mm = err_m.cpu().numpy() * 1000.0
    per_axis = np.sqrt((err_mm ** 2).mean(axis=0))
    overall  = float(np.sqrt((err_mm ** 2).sum(axis=1).mean()))
    return overall, per_axis


def train_one(name: str, model_cls, args, splits, stats,
              train_loader, val_loader, test_loader, device) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = model_cls(in_channels=2, base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    hist_val = []

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for b in train_loader:
            x = b["x"].to(device); y = b["target_pt"].to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        vl = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device); y = b["target_pt"].to(device)
                pred = model(x)
                vl += loss_fn(pred, y).item()
                preds.append(pred); trues.append(y)
        vl /= max(1, len(val_loader))
        hist_val.append(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        sched.step()

        if (ep + 1) % 25 == 0 or ep == args.epochs - 1:
            va_pred = torch.cat(preds, 0); va_true = torch.cat(trues, 0)
            overall, per_axis = rms_mm_per_axis(va_pred, va_true, stats["tgt_std"])
            print(f"  [{name}] ep {ep + 1:3d}/{args.epochs}  "
                  f"val MSE {vl:.4f}  val RMS {overall:5.2f} mm  "
                  f"(X {per_axis[0]:4.2f}, Y {per_axis[1]:4.2f}, Z {per_axis[2]:5.2f})")

    # restore best
    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in test_loader:
            x = b["x"].to(device); y = b["target_pt"].to(device)
            preds.append(model(x)); trues.append(y)
    te_pred = torch.cat(preds, 0); te_true = torch.cat(trues, 0)
    overall, per_axis = rms_mm_per_axis(te_pred, te_true, stats["tgt_std"])

    return {
        "name": name,
        "params_M": n_params / 1e6,
        "best_val_mse": best_val,
        "test_rms_mm": overall,
        "test_rms_per_axis_mm": per_axis.tolist(),
        "history_val_mse": hist_val,
        "train_time_s": time.time() - t0,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[compare] device = {device}")

    import h5py
    with h5py.File(args.data, "r") as f:
        n = f["Q"].shape[0]
    tr, va, te = split_indices(n, seed=args.seed)
    print(f"[compare] splits: train={tr.size} val={va.size} test={te.size}")

    train_ds = ErenPhaseDataset(args.data, indices=tr)
    val_ds   = ErenPhaseDataset(args.data, indices=va, stats=train_ds.stats)
    test_ds  = ErenPhaseDataset(args.data, indices=te, stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    results = []
    for name, cls in FOCUS_ARCH_REGISTRY.items():
        print(f"\n=== training {name} ===")
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        r = train_one(
            name, cls, args, (tr, va, te), train_ds.stats,
            train_loader, val_loader, test_loader, device,
        )
        results.append(r)
        if device == "cuda":
            torch.cuda.empty_cache()

    # ----- pretty summary table -----
    print("\n" + "=" * 80)
    print("FocusPointNet architecture comparison — test split (4 samples)")
    print("=" * 80)
    print(f"{'arch':<8} {'params':>9} {'test RMS':>10} "
          f"{'X (mm)':>8} {'Y (mm)':>8} {'Z (mm)':>8} {'wall (s)':>10}")
    print("-" * 80)
    for r in results:
        pa = r["test_rms_per_axis_mm"]
        print(f"{r['name']:<8} {r['params_M']:>7.3f}M "
              f"{r['test_rms_mm']:>9.2f} "
              f"{pa[0]:>7.2f} {pa[1]:>7.2f} {pa[2]:>7.2f} "
              f"{r['train_time_s']:>9.1f}")

    # save to JSON so the abstract can cite the numbers
    import json
    out_json = args.out_dir / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[compare] saved {out_json}")


if __name__ == "__main__":
    main()
