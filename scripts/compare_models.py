"""
Side-by-side comparison of FNO and U-Net on a shared test split.

Loads both checkpoints (FNO + U-Net, assumed trained on the SAME dataset
with the SAME seed so train/val/test splits match), runs them on the same
test batch, and saves a single figure showing:

    ground truth | FNO prediction | U-Net prediction | FNO error | U-Net error

Usage:
    python scripts/compare_models.py --data data/dataset_v0.h5 \
        --fno outputs/fno_smoke/best.pt --unet outputs/unet_smoke/best.pt \
        --out outputs/comparison.png
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
from unet import UNet2d                                     # noqa: E402

from neuralop.losses import LpLoss                          # noqa: E402
from neuralop.models import FNO                             # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--fno", type=Path, required=True)
    p.add_argument("--unet", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("outputs/comparison.png"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    return p.parse_args()


def split_indices(n, val_frac, test_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(round(test_frac * n))
    n_val  = int(round(val_frac * n))
    return idx[n - n_test :]  # we only need the test indices here


def load_fno(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    m = FNO(
        n_modes=(a["n_modes"], a["n_modes"]),
        in_channels=3,
        out_channels=1,
        hidden_channels=a["hidden_channels"],
        n_layers=a["n_layers"],
        positional_embedding="grid",
    ).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["stats"]


def load_unet(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    m = UNet2d(
        in_channels=3,
        out_channels=1,
        base_channels=a["base_channels"],
    ).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["stats"]


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import h5py
    with h5py.File(args.data, "r") as f:
        n = f["inputs"].shape[0]
    te = split_indices(n, args.val_frac, args.test_frac, args.seed)

    fno, fno_stats = load_fno(args.fno, device)
    unet, unet_stats = load_unet(args.unet, device)

    # use FNO's stats (identical seed => identical split, stats should match;
    # we use the FNO one for denormalisation)
    test_ds = PressureFieldDataset(args.data, indices=te, stats=fno_stats)
    loader = DataLoader(test_ds, batch_size=1)

    loss_fn = LpLoss(d=2, p=2)

    # aggregate metrics across the entire test set
    fno_losses, unet_losses = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            fno_losses.append(loss_fn(fno(x), y).item())
            unet_losses.append(loss_fn(unet(x), y).item())

    fno_mean = float(np.mean(fno_losses))
    unet_mean = float(np.mean(unet_losses))
    print(f"[compare] FNO   test LpLoss = {fno_mean:.4f}")
    print(f"[compare] U-Net test LpLoss = {unet_mean:.4f}")
    print(f"[compare] FNO is {((unet_mean/fno_mean) - 1)*100:+.1f}% "
          f"relative to U-Net")

    # qualitative plot of ONE sample (the median-error FNO case)
    idx_sorted = np.argsort(fno_losses)
    plot_idx = idx_sorted[len(idx_sorted) // 2]

    with torch.no_grad():
        batch = test_ds[plot_idx]
        x = batch["x"].unsqueeze(0).to(device)
        y = batch["y"].unsqueeze(0).to(device)
        fno_pred = fno(x).cpu().numpy()
        unet_pred = unet(x).cpu().numpy()
    y = y.cpu().numpy()

    gt   = test_ds.denormalize_target(y[0, 0])
    fno_p = test_ds.denormalize_target(fno_pred[0, 0])
    un_p  = test_ds.denormalize_target(unet_pred[0, 0])
    fno_err = fno_p - gt
    un_err = un_p - gt

    vmax_p = float(max(gt.max(), fno_p.max(), un_p.max()))
    vmax_e = float(max(np.abs(fno_err).max(), np.abs(un_err).max()))

    fig, ax = plt.subplots(1, 5, figsize=(20, 4.2))
    titles = [
        f"Ground truth\nmax={gt.max():.2e} Pa",
        f"FNO prediction\nmax={fno_p.max():.2e} Pa",
        f"U-Net prediction\nmax={un_p.max():.2e} Pa",
        "FNO error (pred - gt)",
        "U-Net error (pred - gt)",
    ]
    ims = [gt, fno_p, un_p, fno_err, un_err]
    cmaps = ["hot", "hot", "hot", "seismic", "seismic"]
    vlims = [
        (0, vmax_p), (0, vmax_p), (0, vmax_p),
        (-vmax_e, vmax_e), (-vmax_e, vmax_e),
    ]
    for a, im, t, c, (vmin, vmax) in zip(ax, ims, titles, cmaps, vlims):
        h = a.imshow(im, cmap=c, vmin=vmin, vmax=vmax)
        a.set_title(t)
        plt.colorbar(h, ax=a, fraction=0.046)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        f"FNO {fno_mean:.3f}  vs  U-Net {unet_mean:.3f}  "
        f"(LpLoss, lower is better)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[compare] saved {args.out}")


if __name__ == "__main__":
    main()
