"""
Stream Eren's v2 HIFU dataset (the `dataset_cropped/` folder) into ONE
compact HDF5 file. Works with the new layout:

- Individual `.mat` files (MATLAB 5.0, not HDF5) inside a folder
- Each `.mat` contains:
    Q_heat_cropped : (253, 256, 256) float32  -- no NaN, cropped region
    target_pt      : (1, 3)           float64 -- target point in metres
- Phases are NOT embedded in the .mat anymore -- only in the CSV
  (hifu_phase_dataset_1mm.csv, columns Phase_1 .. Phase_256 per ID).

Per sample we:
    1. Load Q_heat_cropped via scipy.io.loadmat
    2. Optional downsample by `--downsample` (default 2 => 127x128x128)
    3. log1p(Q / y_offset) to compress dynamic range
    4. Join with CSV on sim_id -> pull 256 phases, compute sin/cos
    5. Append to HDF5

Usage:
    python scripts/preprocess_eren_v2.py \
        --mat-dir data/eren_new/dataset_cropped \
        --csv hifu_phase_dataset_1mm.csv \
        --out data/eren/dataset_v2.h5 \
        --downsample 2
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.ndimage import zoom

SIM_RE = re.compile(r"sim_id_(\d+)\.mat$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mat-dir", type=Path,
                   default=Path("data/eren_new/dataset_cropped"))
    p.add_argument("--csv", type=Path,
                   default=Path("hifu_phase_dataset_1mm.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("data/eren/dataset_v2.h5"))
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--y-offset", type=float, default=1e3)
    p.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(args.mat_dir.glob("sim_id_*.mat"),
                   key=lambda p: int(SIM_RE.search(p.name).group(1)))
    if not files:
        raise SystemExit(f"no sim_id_*.mat under {args.mat_dir}")

    print(f"[v2] {len(files)} .mat files found in {args.mat_dir}")

    phase_df = pd.read_csv(args.csv).set_index("ID")
    phase_cols = [f"Phase_{i}" for i in range(1, 257)]

    # peek first to learn shapes
    first = loadmat(files[0])
    raw = first["Q_heat_cropped"].astype(np.float32)
    raw_shape = raw.shape
    factor = 1.0 / args.downsample
    if args.downsample != 1:
        probe = zoom(raw, factor, order=1)
        ds_shape = probe.shape
    else:
        ds_shape = raw_shape
    print(f"[v2] raw shape {raw_shape} -> downsampled {ds_shape}  "
          f"(factor={args.downsample})")

    out_dtype = np.float16 if args.dtype == "float16" else np.float32
    total = len(files)

    with h5py.File(args.out, "w") as out_f:
        Q_ds = out_f.create_dataset(
            "Q", (total,) + ds_shape, dtype=out_dtype,
            compression="lzf", chunks=(1,) + ds_shape,
            maxshape=(None,) + ds_shape,
        )
        sincos_ds = out_f.create_dataset(
            "phases_sincos", (total, 2, 256), dtype="float32",
            maxshape=(None, 2, 256),
        )
        phase_raw_ds = out_f.create_dataset(
            "phases_rad", (total, 256), dtype="float32",
            maxshape=(None, 256),
        )
        sim_id_ds = out_f.create_dataset(
            "sim_id", (total,), dtype="int32", maxshape=(None,),
        )
        target_ds = out_f.create_dataset(
            "target_pt_m", (total, 3), dtype="float32",
            maxshape=(None, 3),
        )

        out_f.attrs["raw_shape"] = np.asarray(raw_shape, dtype=np.int32)
        out_f.attrs["ds_shape"] = np.asarray(ds_shape, dtype=np.int32)
        out_f.attrs["downsample"] = int(args.downsample)
        out_f.attrs["y_offset"] = float(args.y_offset)
        out_f.attrs["storage_dtype"] = str(args.dtype)
        out_f.attrs["mat_dir"] = str(args.mat_dir)
        out_f.attrs["csv"] = str(args.csv)
        out_f.attrs["format"] = "v2-cropped-matlab5"

        t0 = time.time()
        written = 0
        missing_csv = 0

        for i, fp in enumerate(files):
            sid = int(SIM_RE.search(fp.name).group(1))

            if sid not in phase_df.index:
                missing_csv += 1
                print(f"    [skip {fp.name}] no CSV row for sim_id={sid}")
                continue

            mat = loadmat(fp)
            Q = mat["Q_heat_cropped"].astype(np.float32)
            tgt = mat["target_pt"].astype(np.float32).squeeze()

            if args.downsample != 1:
                Q_small = zoom(Q, factor, order=1).astype(np.float32)
            else:
                Q_small = Q

            Q_log = np.log1p(np.clip(Q_small, 0, None) / args.y_offset)

            # crop/pad to expected ds_shape (zoom can be off by 1)
            slc = tuple(slice(0, s) for s in ds_shape)
            Q_stored = Q_log[slc].astype(out_dtype)

            ph = phase_df.loc[sid, phase_cols].to_numpy(dtype=np.float64)

            Q_ds[written]          = Q_stored
            sincos_ds[written, 0]  = np.sin(ph).astype(np.float32)
            sincos_ds[written, 1]  = np.cos(ph).astype(np.float32)
            phase_raw_ds[written]  = ph.astype(np.float32)
            sim_id_ds[written]     = sid
            target_ds[written]     = tgt

            written += 1

            if written % 10 == 0 or i == len(files) - 1:
                elapsed = time.time() - t0
                rate = elapsed / max(1, written)
                print(f"[{written:4d}/{total}]  sid={sid:4d}  "
                      f"elapsed={elapsed:5.1f}s  rate={rate:.2f}s/sample")

        if written < total:
            for name in ("Q", "phases_sincos", "phases_rad",
                         "sim_id", "target_pt_m"):
                ds = out_f[name]
                ds.resize((written,) + ds.shape[1:])

    print(f"\n[v2] wrote {written} / {total} samples "
          f"(csv-miss={missing_csv}) to {args.out}")
    print(f"[v2] file size: {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
