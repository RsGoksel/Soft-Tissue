"""
Multi-seed run of the 3 FocusPointNet backbones to obtain statistically
robust per-axis RMS estimates. Runs one (arch, seed) pair at a time, so
GPU memory is bounded by the largest single model, and any crash leaves
a partial table that is still useful.

Produces outputs/focus_arch_compare/multi_seed.json with the per-run
numbers and outputs/focus_arch_compare/multi_seed_summary.md with
mean ± std across seeds.

Usage:
    python scripts/multi_seed_focus.py --seeds 0 1 2 --epochs 120
"""
from __future__ import annotations

import argparse
import json
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
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "outputs" / "focus_arch_compare")
    return p.parse_args()


def split_indices(n: int, seed: int, val_frac=0.15, test_frac=0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    n_val  = max(1, int(round(val_frac  * n)))
    return (idx[: n - n_val - n_test],
            idx[n - n_val - n_test : n - n_test],
            idx[n - n_test :])


def rms_mm_per_axis(pred_std, true_std, tgt_std):
    std_t = torch.as_tensor(tgt_std, dtype=pred_std.dtype, device=pred_std.device)
    err_mm = (pred_std - true_std).cpu().numpy() * tgt_std * 1000.0
    per_axis = np.sqrt((err_mm ** 2).mean(axis=0))
    overall  = float(np.sqrt((err_mm ** 2).sum(axis=1).mean()))
    return overall, per_axis


def train_once(name, cls, args, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)

    import h5py
    with h5py.File(args.data, "r") as f:
        n = f["Q"].shape[0]
    tr, va, te = split_indices(n, seed=seed)

    train_ds = ErenPhaseDataset(args.data, indices=tr)
    val_ds   = ErenPhaseDataset(args.data, indices=va, stats=train_ds.stats)
    test_ds  = ErenPhaseDataset(args.data, indices=te, stats=train_ds.stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    model = cls(in_channels=2, base_channels=args.base_channels).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_state = None
    for ep in range(args.epochs):
        model.train()
        for b in train_loader:
            x = b["x"].to(device); y = b["target_pt"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for b in val_loader:
                x = b["x"].to(device); y = b["target_pt"].to(device)
                vl += loss_fn(model(x), y).item()
        vl /= max(1, len(val_loader))
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sched.step()

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for b in test_loader:
            x = b["x"].to(device); y = b["target_pt"].to(device)
            preds.append(model(x)); trues.append(y)
    te_pred = torch.cat(preds, 0); te_true = torch.cat(trues, 0)
    overall, per_axis = rms_mm_per_axis(te_pred, te_true, train_ds.stats["tgt_std"])

    return {
        "name": name, "seed": int(seed),
        "best_val_mse": float(best_val),
        "test_rms_mm": float(overall),
        "test_rms_x": float(per_axis[0]),
        "test_rms_y": float(per_axis[1]),
        "test_rms_z": float(per_axis[2]),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[multi-seed] device={device}  seeds={args.seeds}  "
          f"epochs={args.epochs}")

    results = []
    t_start = time.time()
    for seed in args.seeds:
        for name, cls in FOCUS_ARCH_REGISTRY.items():
            print(f"\n  [run] seed={seed}  arch={name} ...")
            if device == "cuda":
                torch.cuda.empty_cache()
            t0 = time.time()
            try:
                r = train_once(name, cls, args, seed, device)
                r["wall_s"] = time.time() - t0
                print(f"    -> test RMS {r['test_rms_mm']:.2f} mm  "
                      f"(X {r['test_rms_x']:.2f}, Y {r['test_rms_y']:.2f}, "
                      f"Z {r['test_rms_z']:.2f})  wall {r['wall_s']:.0f}s")
            except RuntimeError as e:
                print(f"    CRASH: {e}")
                r = {"name": name, "seed": int(seed), "error": str(e)}
            results.append(r)

            # incremental save after every run (partial-result safety)
            with open(args.out_dir / "multi_seed.json", "w",
                      encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    # summary by arch
    by_arch = {n: [] for n in FOCUS_ARCH_REGISTRY}
    for r in results:
        if "error" not in r:
            by_arch[r["name"]].append(r)

    lines = ["# Multi-seed FocusPointNet comparison\n",
             f"- epochs per run: {args.epochs}",
             f"- seeds: {args.seeds}",
             f"- batch size: {args.batch_size}",
             f"- base_channels: {args.base_channels}",
             f"- total wall time: {(time.time() - t_start)/60:.1f} min\n",
             "| arch | n_runs | test RMS mean ± std | X mean ± std | "
             "Y mean ± std | Z mean ± std |",
             "|------|--------|---------------------|--------------|"
             "--------------|--------------|"]

    for arch, rs in by_arch.items():
        if not rs:
            continue
        tot = np.array([r["test_rms_mm"] for r in rs])
        xs  = np.array([r["test_rms_x"]  for r in rs])
        ys  = np.array([r["test_rms_y"]  for r in rs])
        zs  = np.array([r["test_rms_z"]  for r in rs])
        lines.append(
            f"| {arch} | {len(rs)} | "
            f"{tot.mean():.2f} ± {tot.std():.2f} | "
            f"{xs.mean():.2f} ± {xs.std():.2f} | "
            f"{ys.mean():.2f} ± {ys.std():.2f} | "
            f"{zs.mean():.2f} ± {zs.std():.2f} |"
        )

    (args.out_dir / "multi_seed_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\n[multi-seed] wrote {args.out_dir / 'multi_seed_summary.md'}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
