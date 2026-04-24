"""
Train the inverse 3D CNN on Eren's HIFU dataset.

Input  : (2, D, H, W) -- (log_Q normalised, mask)
Output : (2, 256)      -- (sin, cos) transducer phases

Usage:
    python scripts/train_eren_inverse.py --data data/eren/dataset.h5 --epochs 80
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

from eren_dataset import ErenPhaseDataset                                   # noqa: E402
from eren_model import (                                                    # noqa: E402
    CircularPhaseLoss,
    GaugeInvariantPhaseLoss,
    PhaseInverseNet,
    SinCosMSELoss,
    phase_error_degrees,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/eren/dataset.h5"))
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=16,
                   help="U-Net-ish width; 16 keeps the model on an 8GB GPU")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--loss", choices=["sincos", "circular", "gauge"],
                   default="gauge",
                   help="gauge: invariant to global phase shift (recommended "
                        "for HIFU - Q does not constrain the global phase). "
                        "circular: per-element 1-cos. sincos: plain MSE.")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/eren_inverse"))
    p.add_argument("--use-target-pt", action="store_true",
                   help="concat target_pt (3,) to the MLP bottleneck")
    return p.parse_args()


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
    print(f"[eren] device={device}")

    import h5py
    with h5py.File(args.data, "r") as f:
        n = f["Q"].shape[0]
    tr, va, te = split_indices(n, args.val_frac, args.test_frac, args.seed)
    print(f"[eren] train={tr.size} val={va.size} test={te.size}")

    train_ds = ErenPhaseDataset(args.data, indices=tr)
    val_ds   = ErenPhaseDataset(args.data, indices=va, stats=train_ds.stats)
    test_ds  = ErenPhaseDataset(args.data, indices=te, stats=train_ds.stats)
    print(f"[eren] q_max={train_ds.stats['q_max']:.3f}  "
          f"ds_shape={train_ds.ds_shape}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    model = PhaseInverseNet(
        in_channels=2,
        n_transducers=256,
        base_channels=args.base_channels,
        use_target_pt=args.use_target_pt,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eren] PhaseInverseNet params: {n_params / 1e6:.2f}M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    if args.loss == "sincos":
        loss_fn = SinCosMSELoss()
    elif args.loss == "circular":
        loss_fn = CircularPhaseLoss()
    else:
        loss_fn = GaugeInvariantPhaseLoss()

    def forward(batch):
        x = batch["x"].to(device)
        tgt = (batch["target_pt"].to(device)
               if args.use_target_pt and "target_pt" in batch
               else None)
        return model(x, target_pt=tgt) if args.use_target_pt else model(x)

    history = {"train": [], "val_loss": [], "val_deg": []}
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            y = batch["y"].to(device)
            optim.zero_grad()
            pred = forward(batch)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        va_loss = 0.0
        va_deg = 0.0
        with torch.no_grad():
            for batch in val_loader:
                y = batch["y"].to(device)
                pred = forward(batch)
                va_loss += loss_fn(pred, y).item()
                va_deg += phase_error_degrees(pred, y).item()
        va_loss /= max(1, len(val_loader))
        va_deg  /= max(1, len(val_loader))

        sched.step()
        history["train"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_deg"].append(va_deg)
        print(f"epoch {epoch + 1:3d}/{args.epochs}  "
              f"train {tr_loss:.4f}  val {va_loss:.4f}  "
              f"val phase err {va_deg:5.1f} deg")

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

    # test
    ckpt = torch.load(args.out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    te_loss = 0.0
    te_deg = 0.0
    with torch.no_grad():
        for batch in test_loader:
            y = batch["y"].to(device)
            pred = forward(batch)
            te_loss += loss_fn(pred, y).item()
            te_deg += phase_error_degrees(pred, y).item()
    te_loss /= max(1, len(test_loader))
    te_deg  /= max(1, len(test_loader))
    print(f"[eren] TEST loss {te_loss:.4f}  phase err {te_deg:5.1f} deg")

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel(f"{args.loss} loss"); ax[0].legend()
    ax[0].set_title("Training loss")
    ax[1].plot(history["val_deg"], color="tab:red")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mean phase error [deg]")
    ax[1].set_title("Val phase error")
    fig.tight_layout()
    fig.savefig(args.out_dir / "loss_curve.png", dpi=120)

    # sample prediction sanity figure
    with torch.no_grad():
        batch = next(iter(test_loader))
        y = batch["y"].to(device)
        pred = forward(batch).cpu().numpy()
    y = y.cpu().numpy()
    pred_phi = np.arctan2(pred[0, 0], pred[0, 1])
    true_phi = np.arctan2(y[0, 0], y[0, 1])
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(true_phi, "o-", label="target", alpha=0.7, markersize=3)
    ax.plot(pred_phi, "x-", label="predicted", alpha=0.7, markersize=3)
    ax.set_xlabel("transducer index")
    ax.set_ylabel("phase [rad]")
    ax.legend()
    ax.set_title("Target vs predicted transducer phases (1 test sample)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "phase_sample.png", dpi=120)

    print(f"[eren] saved checkpoint + figures to {args.out_dir}")


if __name__ == "__main__":
    main()
