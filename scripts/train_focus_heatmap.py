"""
Train HeatmapUNet3D / HeatmapUNet3DWithOffset on the Q -> focus-point task.

The heatmap pipeline:
  1. Fit an affine voxel-from-target map on TRAINING samples using the
     per-sample weighted centroid of Q as a proxy (classical heatmap-based
     landmark setup; Payer et al. 2019, Weihsbach et al. 2025).
  2. Build per-sample Gaussian target heatmaps at the corresponding voxel.
  3. MSE loss on the heatmap (+ optional offset head).
  4. Inference: soft-argmax predicted heatmap -> voxel -> inverse affine
     -> predicted target_pt_m, compared against ground truth in mm.

Usage:
    python scripts/train_focus_heatmap.py --arch heatmap --epochs 120
    python scripts/train_focus_heatmap.py --arch heatmap_offset --epochs 120
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eren_dataset import ErenPhaseDataset                    # noqa: E402
from focus_heatmap import (                                   # noqa: E402
    FOCUS_HEATMAP_REGISTRY,
    make_gaussian_heatmap,
    soft_argmax_3d,
    fit_voxel_affine,
    voxel_from_target,
    target_from_voxel,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path,
                   default=PROJECT_ROOT / "data" / "eren" / "dataset_v2.h5")
    p.add_argument("--arch", choices=list(FOCUS_HEATMAP_REGISTRY),
                   default="heatmap")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--sigma", type=float, default=6.0,
                   help="Gaussian target width, in voxels.")
    p.add_argument("--loss", choices=["dsnt", "weighted_mse"], default="dsnt",
                   help="dsnt: supervise the soft-argmax coordinate directly "
                        "(DSNT, Nibali et al. 2018) -- robust on 30-sample "
                        "volumes, does not collapse. weighted_mse: MSE on the "
                        "heatmap itself, emphasising target voxels.")
    p.add_argument("--loss-weight-eps", type=float, default=0.01,
                   help="Per-voxel loss weight floor for weighted_mse.")
    p.add_argument("--variance-reg", type=float, default=1e-4,
                   help="Variance regulariser on the predicted heatmap for "
                        "DSNT loss -- keeps the distribution peaky so the "
                        "soft-argmax is crisp.")
    p.add_argument("--offset-weight", type=float, default=0.1,
                   help="weight of offset-head L1 loss if applicable.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "outputs" / "focus_heatmap")
    return p.parse_args()


def split_indices(n: int, seed: int, val_frac=0.15, test_frac=0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(test_frac * n)))
    n_val  = max(1, int(round(val_frac  * n)))
    return (idx[: n - n_val - n_test],
            idx[n - n_val - n_test : n - n_test],
            idx[n - n_test :])


def focal_voxel(Q: np.ndarray) -> np.ndarray:
    """
    (N, D, H, W) Q -> (N, 3) focal voxel in (z, y, x).

    We use a *high-threshold weighted centroid* of the expm1-decoded Q:
    voxels with Q > 0.85 * Qmax only contribute. This isolates the focal
    spot and ignores the long pre-focal beam path that otherwise biases
    a plain centroid of Q along the depth axis. Much more reliable
    affine-fit anchor than either plain centroid (diffuse) or argmax
    (noisy single voxel).
    """
    out = np.zeros((Q.shape[0], 3), dtype=np.float32)
    zz, yy, xx = None, None, None
    for i in range(Q.shape[0]):
        v = np.expm1(Q[i].astype(np.float64))
        v = np.clip(v, 0, None)
        thr = 0.85 * v.max()
        m = v >= thr
        w = np.where(m, v, 0.0)
        s = w.sum()
        if s <= 0:
            # fallback to argmax if masking emptied the volume
            k = int(Q[i].argmax())
            out[i] = np.unravel_index(k, Q[i].shape)
            continue
        if zz is None:
            zz, yy, xx = np.meshgrid(
                np.arange(v.shape[0]),
                np.arange(v.shape[1]),
                np.arange(v.shape[2]),
                indexing="ij",
            )
        out[i, 0] = (zz * w).sum() / s
        out[i, 1] = (yy * w).sum() / s
        out[i, 2] = (xx * w).sum() / s
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"[heatmap] arch={args.arch}  device={device}  seed={args.seed}")

    with h5py.File(args.data, "r") as f:
        n = f["Q"].shape[0]
    tr_idx, va_idx, te_idx = split_indices(n, seed=args.seed)
    print(f"[heatmap] train={tr_idx.size} val={va_idx.size} test={te_idx.size}")

    train_ds = ErenPhaseDataset(args.data, indices=tr_idx)
    val_ds   = ErenPhaseDataset(args.data, indices=va_idx, stats=train_ds.stats)
    test_ds  = ErenPhaseDataset(args.data, indices=te_idx, stats=train_ds.stats)

    # ---- fit voxel-from-target affine on TRAINING split ----
    tgt_mean = torch.as_tensor(train_ds.stats["tgt_mean"], dtype=torch.float32)
    tgt_std  = torch.as_tensor(train_ds.stats["tgt_std"],  dtype=torch.float32)

    with h5py.File(args.data, "r") as f:
        Q_train = np.asarray(f["Q"][sorted(tr_idx.tolist())], dtype=np.float32)
        tgt_train_m = np.asarray(f["target_pt_m"][sorted(tr_idx.tolist())],
                                 dtype=np.float32)
    focal_train = focal_voxel(Q_train)
    fit = fit_voxel_affine(
        torch.as_tensor(tgt_train_m, dtype=torch.float32),
        torch.as_tensor(focal_train, dtype=torch.float32),
    )
    # sanity: verify the fit actually inverts well on training split
    fit_tgt_tensor = torch.as_tensor(tgt_train_m, dtype=torch.float32)
    fit_vox_tensor = torch.as_tensor(focal_train, dtype=torch.float32)
    pred_vox = voxel_from_target(fit_tgt_tensor, fit).numpy()
    fit_rms = np.sqrt(((pred_vox - focal_train) ** 2).sum(axis=1).mean())
    print(f"[heatmap] affine fit train-voxel RMS: {fit_rms:.2f} voxels "
          f"(spacing 2 mm -> {fit_rms * 2:.1f} mm). "
          f"Good <6 voxels (<12 mm), poor >15 voxels.")
    print(f"[heatmap] fit perm={fit['perm'].tolist()}  "
          f"slope={fit['slope'].tolist()}  offset={fit['offset'].tolist()}")

    ds_shape = tuple(map(int, train_ds.ds_shape))
    print(f"[heatmap] volume shape {ds_shape}  sigma={args.sigma} voxels")

    model = FOCUS_HEATMAP_REGISTRY[args.arch](
        in_channels=2, base_channels=args.base_channels
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[heatmap] {args.arch} params: {n_params/1e6:.3f}M")

    has_offset = args.arch == "heatmap_offset"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def voxel_targets(tgt_std_batch: torch.Tensor) -> torch.Tensor:
        # un-standardise, apply affine -> (B, 3) in (z, y, x) voxels
        tgt_m = tgt_std_batch * tgt_std.to(tgt_std_batch.device) \
                + tgt_mean.to(tgt_std_batch.device)
        return voxel_from_target(tgt_m, fit)

    # weighted-MSE helper -- kept as a fallback option.
    def weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        w = target + args.loss_weight_eps
        return (w * (pred - target) ** 2).sum() / w.sum()

    def dsnt_loss(pred_hm: torch.Tensor,
                  target_voxels: torch.Tensor) -> torch.Tensor:
        """DSNT: differentiable spatial-to-numerical transform (Nibali 2018).
        Instead of MSE on the heatmap itself, supervise the expected
        voxel coordinate of the predicted heatmap (via soft-argmax) so
        the gradient directly points toward the target location. A small
        variance regulariser keeps the distribution peaky."""
        coords = soft_argmax_3d(pred_hm)                 # (B, 3) in voxels
        coord_loss = F.mse_loss(coords, target_voxels)
        # variance regulariser: penalise large spread of the softmax'd heatmap
        B, _, D, H, W = pred_hm.shape
        flat = pred_hm.reshape(B, -1)
        prob = F.softmax(flat, dim=1).reshape(B, 1, D, H, W)
        zz = torch.arange(D, device=pred_hm.device,
                          dtype=pred_hm.dtype)[None, None, :, None, None]
        yy = torch.arange(H, device=pred_hm.device,
                          dtype=pred_hm.dtype)[None, None, None, :, None]
        xx = torch.arange(W, device=pred_hm.device,
                          dtype=pred_hm.dtype)[None, None, None, None, :]
        mz = (prob * zz).sum(dim=(2, 3, 4))              # (B, 1)
        my = (prob * yy).sum(dim=(2, 3, 4))
        mx = (prob * xx).sum(dim=(2, 3, 4))
        var_z = (prob * (zz - mz[..., None, None]) ** 2).sum(dim=(2, 3, 4))
        var_y = (prob * (yy - my[..., None, None]) ** 2).sum(dim=(2, 3, 4))
        var_x = (prob * (xx - mx[..., None, None]) ** 2).sum(dim=(2, 3, 4))
        var_reg = (var_z + var_y + var_x).mean()
        return coord_loss + args.variance_reg * var_reg

    def step(batch, train: bool):
        x = batch["x"].to(device)
        y = batch["target_pt"].to(device)
        v = voxel_targets(y)                       # (B, 3)
        if has_offset:
            pred_hm, pred_off = model(x)
            if args.loss == "dsnt":
                loss_hm = dsnt_loss(pred_hm, v)
            else:
                target_hm = make_gaussian_heatmap(v, ds_shape, sigma=args.sigma)
                loss_hm = weighted_mse(pred_hm, target_hm)
            # offset loss only at target voxel location
            with torch.no_grad():
                vi = v.round().long().clamp(min=torch.zeros(1, device=device, dtype=torch.long),
                                            max=torch.tensor(list(ds_shape), device=device) - 1)
            B = x.size(0)
            off_at = torch.stack([
                pred_off[b, :, vi[b, 0], vi[b, 1], vi[b, 2]] for b in range(B)
            ], dim=0)                                 # (B, 3)
            residual = (v - vi.float())               # sub-voxel remainder
            loss_off = F.l1_loss(off_at, residual)
            loss = loss_hm + args.offset_weight * loss_off
            return loss, pred_hm, pred_off, v
        pred_hm = model(x)
        if args.loss == "dsnt":
            loss = dsnt_loss(pred_hm, v)
        else:
            target_hm = make_gaussian_heatmap(v, ds_shape, sigma=args.sigma)
            loss = weighted_mse(pred_hm, target_hm)
        return loss, pred_hm, None, v

    def rms_mm(v_pred: torch.Tensor, v_true: torch.Tensor) -> tuple[float, np.ndarray]:
        # voxel -> target_pt_m via inverse affine
        tgt_pred_m = target_from_voxel(v_pred, fit)
        tgt_true_m = target_from_voxel(v_true, fit)
        err = (tgt_pred_m - tgt_true_m).detach().cpu().numpy() * 1000.0
        per_axis = np.sqrt((err ** 2).mean(axis=0))
        overall  = float(np.sqrt((err ** 2).sum(axis=1).mean()))
        return overall, per_axis

    def evaluate(loader) -> tuple[float, np.ndarray, float]:
        model.eval()
        total_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for b in loader:
                loss, pred_hm, pred_off, v_true = step(b, train=False)
                total_loss += loss.item()
                # soft-argmax predicted voxel
                v_pred = soft_argmax_3d(pred_hm)
                if pred_off is not None:
                    B = v_pred.size(0)
                    vi = v_pred.round().long().clamp(
                        min=torch.zeros(1, device=device, dtype=torch.long),
                        max=torch.tensor(list(ds_shape), device=device) - 1,
                    )
                    off = torch.stack([
                        pred_off[i, :, vi[i, 0], vi[i, 1], vi[i, 2]]
                        for i in range(B)
                    ], dim=0)
                    v_pred = v_pred + off
                preds.append(v_pred); trues.append(v_true)
        pred_v = torch.cat(preds, 0); true_v = torch.cat(trues, 0)
        rms, per_axis = rms_mm(pred_v, true_v)
        return rms, per_axis, total_loss / max(1, len(loader))

    best_val = float("inf")
    best_state = None
    history = {"val_loss": [], "val_rms_mm": []}
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for b in train_loader:
            loss, *_ = step(b, train=True)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        va_rms, va_pa, va_loss = evaluate(val_loader)
        history["val_loss"].append(va_loss)
        history["val_rms_mm"].append(va_rms)
        sched.step()

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep+1:3d}/{args.epochs}  train {tr_loss:.4f}  "
                  f"val loss {va_loss:.4f}  val RMS {va_rms:5.2f} mm  "
                  f"(X {va_pa[0]:4.2f}, Y {va_pa[1]:4.2f}, Z {va_pa[2]:5.2f})")

    model.load_state_dict(best_state)
    te_rms, te_pa, te_loss = evaluate(test_loader)
    print(f"\n[heatmap] TEST  loss {te_loss:.4f}  RMS {te_rms:.2f} mm  "
          f"(X {te_pa[0]:.2f}, Y {te_pa[1]:.2f}, Z {te_pa[2]:.2f})")

    out = {
        "arch": args.arch, "seed": args.seed, "epochs": args.epochs,
        "params_M": n_params / 1e6,
        "test_rms_mm": te_rms,
        "test_rms_x":  float(te_pa[0]),
        "test_rms_y":  float(te_pa[1]),
        "test_rms_z":  float(te_pa[2]),
        "wall_s": time.time() - t0,
        "fit_slope":  [float(s) for s in fit["slope"].tolist()],
        "fit_offset": [float(s) for s in fit["offset"].tolist()],
        "fit_perm":   [int(s) for s in fit["perm"].tolist()],
    }
    with open(args.out_dir / f"results_{args.arch}_seed{args.seed}.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    torch.save({"model": best_state, "fit": fit, "args": vars(args)},
               args.out_dir / f"best_{args.arch}_seed{args.seed}.pt")
    print(f"[heatmap] saved to {args.out_dir}")


if __name__ == "__main__":
    main()
