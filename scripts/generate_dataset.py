"""
Batch dataset generator -- NATIVE path over OpenBreastUS phantoms.

For each sample:
    1. pick a random OpenBreastUS phantom (resampled to `grid`)
    2. insert a random tumor into the continuous sound-speed map
    3. run k-wave-python with `run_focused_sim_from_speed`
    4. store (c, rho, alpha) as inputs and p_max as target in an HDF5 file

Usage:
    python scripts/generate_dataset.py --n 200 --out data/dataset_v0.h5
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phantom import insert_tumor_speed, load_openbreastus_speedmaps   # noqa: E402
from scipy.ndimage import zoom                                         # noqa: E402
from simulate import SimConfig, run_focused_sim_from_speed             # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mat", type=Path,
                   default=Path("data/breast_train_speed.mat"),
                   help="OpenBreastUS .mat file to pull phantoms from")
    p.add_argument("--n", type=int, default=200, help="number of samples")
    p.add_argument("--grid", type=int, default=256,
                   help="resample phantom to this side length")
    p.add_argument("--out", type=Path, default=Path("data/dataset_v0.h5"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    cfg = SimConfig(use_gpu=args.gpu)

    print(f"[gen] loading OpenBreastUS volume: {args.mat}")
    vol = load_openbreastus_speedmaps(args.mat)
    print(f"[gen] volume shape={vol.shape}, value range "
          f"{vol.min():.1f}..{vol.max():.1f} m/s")

    n_phantoms = vol.shape[0]
    native_side = vol.shape[1]
    resample_factor = args.grid / native_side

    # pick a random subset of phantom indices (with replacement if n > pool)
    if args.n <= n_phantoms:
        phantom_indices = rng.choice(n_phantoms, size=args.n, replace=False)
    else:
        phantom_indices = rng.integers(0, n_phantoms, size=args.n)

    # output shape is padded by the water standoff inside the sim; we query
    # the padding amount by running a dummy conversion
    pad = int(round(cfg.water_standoff_mm * 1e-3 / cfg.dx))
    H = args.grid + pad
    W = args.grid
    print(f"[gen] output grid: {H} axial x {W} lateral  (pad={pad})")

    with h5py.File(args.out, "w") as f:
        X = f.create_dataset(
            "inputs",  (args.n, 3, H, W), dtype="float32",
            maxshape=(None, 3, H, W),
        )
        Y = f.create_dataset(
            "targets", (args.n, 1, H, W), dtype="float32",
            maxshape=(None, 1, H, W),
        )
        F = f.create_dataset(
            "focus", (args.n, 2), dtype="int32", maxshape=(None, 2),
        )
        P = f.create_dataset(
            "phantom_index", (args.n,), dtype="int32", maxshape=(None,),
        )

        f.attrs["source_f0_Hz"] = cfg.source_f0
        f.attrs["dx_m"] = cfg.dx
        f.attrs["source_amp_Pa"] = cfg.source_amp
        f.attrs["water_standoff_mm"] = cfg.water_standoff_mm
        f.attrs["mat_file"] = str(args.mat)
        f.attrs["grid"] = args.grid

        t0 = time.time()
        n_written = 0
        for i, pidx in enumerate(phantom_indices):
            # 1) load + resize phantom
            speed = vol[int(pidx)].astype(np.float32)
            if resample_factor != 1.0:
                speed = zoom(speed, zoom=resample_factor, order=1).astype(np.float32)

            # 2) insert random tumor
            try:
                speed_with_tumor, (cy, cx, _r) = insert_tumor_speed(
                    speed, rng=rng,
                )
            except (ValueError, RuntimeError) as e:
                print(f"[skip {i}] tumor placement failed: {e}")
                continue

            # 3) run k-wave
            try:
                res = run_focused_sim_from_speed(
                    speed_with_tumor,
                    focus_yx=(cy, cx),
                    config=cfg,
                )
            except Exception as e:
                print(f"[skip {i}] sim failed: {e}")
                continue

            # 4) write (reject samples whose padded shape drifted)
            if res["c"].shape != (H, W):
                print(f"[skip {i}] shape mismatch {res['c'].shape} != {(H, W)}")
                continue

            X[n_written, 0] = res["c"]
            X[n_written, 1] = res["rho"]
            X[n_written, 2] = res["alpha"]
            Y[n_written, 0] = res["p_max"]
            F[n_written]    = res["focus_yx"]
            P[n_written]    = int(pidx)
            n_written += 1

            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            eta = rate * (args.n - i - 1)
            print(f"[{i + 1:4d}/{args.n}]  phantom={int(pidx):5d}  "
                  f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")

        # trim datasets to the actually-written count
        if n_written < args.n:
            print(f"[gen] trimming datasets from {args.n} to {n_written}")
            for ds_name in ("inputs", "targets", "focus", "phantom_index"):
                ds = f[ds_name]
                new_shape = (n_written,) + ds.shape[1:]
                ds.resize(new_shape)

    print(f"\n[gen] done. wrote {n_written} samples to {args.out}")


if __name__ == "__main__":
    main()
