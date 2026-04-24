"""
Train a 2D Fourier Neural Operator on the HDF5 dataset.

Usage:
    python scripts/train_fno.py --data data/dataset_v0.h5 --epochs 50

Model maps (c, rho, alpha) -> peak pressure. All channels are normalized in
the Dataset wrapper. LpLoss is used as training + eval loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import PressureFieldDataset                    # noqa: E402

from neuralop.losses import H1Loss, LpLoss                  # noqa: E402
from neuralop.models import FNO                             # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-modes", type=int, default=32)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/fno_run"))
    p.add_argument("--loss", choices=["lp", "h1", "lp+h1"], default="lp+h1",
                   help="training loss: LpLoss, H1Loss, or weighted sum")
    p.add_argument("--h1-weight", type=float, default=0.3,
                   help="weight of H1 term when --loss=lp+h1")
    p.add_argument("--no-log-target", action="store_true",
                   help="disable log-space target scaling")
    return p.parse_args()


class CombinedLoss:
    """LpLoss + w * H1Loss -- encourages both pointwise and gradient
    fidelity, which matters for interference patterns."""

    def __init__(self, d: int = 2, h1_weight: float = 0.3) -> None:
        self.lp = LpLoss(d=d, p=2)
        self.h1 = H1Loss(d=d)
        self.w = h1_weight

    def __call__(self, pred, y):
        return self.lp(pred, y) + self.w * self.h1(pred, y)


def split_indices(n: int, val_frac: float, test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(test_frac * n))
    n_val  = int(round(val_frac * n))
    return (
        idx[: n - n_val - n_test],
        idx[n - n_val - n_test : n - n_test],
        idx[n - n_test :],
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_fno] device={device}")

    # ---- data ----
    import h5py
    with h5py.File(args.data, "r") as f:
        n = f["inputs"].shape[0]
    tr, va, te = split_indices(n, args.val_frac, args.test_frac, args.seed)
    print(f"[train_fno] train={tr.size} val={va.size} test={te.size}")

    log_target = not args.no_log_target
    train_ds = PressureFieldDataset(args.data, indices=tr, log_target=log_target)
    val_ds   = PressureFieldDataset(args.data, indices=va, stats=train_ds.stats)
    test_ds  = PressureFieldDataset(args.data, indices=te, stats=train_ds.stats)
    print(f"[train_fno] log_target={log_target}")
    if log_target:
        print(f"[train_fno] log_scale={train_ds.stats['log_scale']:.3f}  "
              f"y_offset={train_ds.stats['y_offset']:.1e}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # ---- model ----
    model = FNO(
        n_modes=(args.n_modes, args.n_modes),
        in_channels=3,
        out_channels=1,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        positional_embedding="grid",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_fno] FNO params: {n_params / 1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    if args.loss == "lp":
        loss_fn = LpLoss(d=2, p=2)
    elif args.loss == "h1":
        loss_fn = H1Loss(d=2)
    else:
        loss_fn = CombinedLoss(d=2, h1_weight=args.h1_weight)
    eval_fn = LpLoss(d=2, p=2)  # report Lp for easy comparison
    print(f"[train_fno] loss={args.loss}")

    # ---- train loop ----
    history = {"train": [], "val": []}
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optim.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                pred = model(x)
                va_loss += eval_fn(pred, y).item()
        va_loss /= max(1, len(val_loader))

        sched.step()
        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  train {tr_loss:.4f}  val {va_loss:.4f}")

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

    # ---- final test + qualitative figure ----
    ckpt = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    te_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            te_loss += eval_fn(pred, y).item()
    te_loss /= max(1, len(test_loader))
    print(f"[train_fno] TEST LpLoss: {te_loss:.4f}")

    # plot one test sample, denormalized to physical units (Pa)
    with torch.no_grad():
        batch = next(iter(test_loader))
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        pred = model(x).cpu().numpy()
    y = y.cpu().numpy()

    gt_phys = test_ds.denormalize_target(y[0, 0])
    pd_phys = test_ds.denormalize_target(pred[0, 0])
    err_phys = pd_phys - gt_phys
    vmax_pres = float(max(gt_phys.max(), pd_phys.max()))
    vmax_err = float(np.abs(err_phys).max())
    rel = (np.abs(err_phys).mean() / (np.abs(gt_phys).mean() + 1e-8)) * 100

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    im0 = ax[0].imshow(gt_phys, cmap="hot", vmin=0, vmax=vmax_pres)
    ax[0].set_title(f"Ground truth\nmax={gt_phys.max():.2e} Pa")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(pd_phys, cmap="hot", vmin=0, vmax=vmax_pres)
    ax[1].set_title(f"FNO prediction\nmax={pd_phys.max():.2e} Pa")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(err_phys, cmap="seismic", vmin=-vmax_err, vmax=vmax_err)
    ax[2].set_title(f"Error (pred - gt)\nrel L1 ≈ {rel:.1f}%")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out_dir / "test_sample.png", dpi=120)

    # loss curves
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("LpLoss"); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "loss_curve.png", dpi=120)

    print(f"[train_fno] saved checkpoint + figures to {args.out_dir}")


if __name__ == "__main__":
    main()
