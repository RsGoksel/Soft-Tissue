"""
Stream Eren's HIFU dataset from the 5 zip archives directly into ONE compact
HDF5 file. Never extracts the zips to disk -- avoids running out of space on
C: (9 GB zips -> 9 GB extracted -> too much).

Per sample we:
    1. Open sim_id_XXXX.mat from inside the zip as an in-memory buffer
    2. Read Q_heat (310, 321, 253) float32 + phases (1, 256) float64
    3. Replace NaN with 0 and produce a valid_mask
    4. Downsample Q by 2x on every axis (linear) -> ~(155, 161, 127)
    5. Apply log1p(Q / y_offset) to compress 4-5 decades dynamic range
    6. Write to data/eren/dataset.h5

Usage:
    python scripts/preprocess_eren.py \
        --zip-glob "dataset-20260409T142546Z-3-*.zip" \
        --csv hifu_phase_dataset_1mm.csv \
        --out data/eren/dataset.h5
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import zoom


SIM_RE = re.compile(r"sim_id_(\d+)\.mat$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--zip-glob", type=str, required=True,
                   help="glob for the 5 dataset zip files")
    p.add_argument("--csv", type=Path, required=True,
                   help="hifu_phase_dataset_1mm.csv")
    p.add_argument("--out", type=Path, default=Path("data/eren/dataset.h5"))
    p.add_argument("--downsample", type=int, default=2,
                   help="spatial downsample factor applied to Q_heat")
    p.add_argument("--y-offset", type=float, default=1e3,
                   help="floor value for Q log1p normalisation (W/m^3)")
    p.add_argument("--dtype", choices=["float32", "float16"], default="float16",
                   help="on-disk dtype for Q (float16 halves storage)")
    return p.parse_args()


def iter_mat_files(zip_paths: list[Path]):
    """Yield (sim_id, bytes) for every sim_id_XXXX.mat found in any zip,
    in ascending sim_id order."""
    index: dict[int, tuple[Path, str]] = {}
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                m = SIM_RE.search(info.filename)
                if m:
                    sid = int(m.group(1))
                    index[sid] = (zp, info.filename)

    for sid in sorted(index.keys()):
        zp, name = index[sid]
        with zipfile.ZipFile(zp) as zf:
            with zf.open(name) as member:
                yield sid, member.read()


def load_mat_from_bytes(buf: bytes) -> dict:
    bio = io.BytesIO(buf)
    with h5py.File(bio, "r") as f:
        out = {
            "Q_heat": f["Q_heat"][...].astype(np.float32),
            "phases": f["phases"][...].astype(np.float64).squeeze(),
            "target_pt": f["target_pt"][...].astype(np.float32).squeeze(),
        }
    return out


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(Path(p) for p in glob(args.zip_glob))
    if not zip_paths:
        raise SystemExit(f"no zips match {args.zip_glob}")
    print(f"[preproc] found {len(zip_paths)} zips:")
    for zp in zip_paths:
        print(f"    {zp.name}")

    phase_df = pd.read_csv(args.csv).set_index("ID")
    phase_cols = [f"Phase_{i}" for i in range(1, 257)]

    # peek the first .mat to learn the downsampled shape
    first_iter = iter_mat_files(zip_paths)
    sid0, buf0 = next(first_iter)
    sample0 = load_mat_from_bytes(buf0)
    raw_shape = sample0["Q_heat"].shape
    ds_shape = tuple(
        max(1, int(round(s / args.downsample))) for s in raw_shape
    )
    print(f"[preproc] raw Q shape {raw_shape} -> downsampled {ds_shape}")
    del sample0, buf0

    out_dtype = np.float16 if args.dtype == "float16" else np.float32

    # now do a proper counting pass (cheap, only reads zip indices)
    total = 0
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            total += sum(1 for i in zf.infolist() if SIM_RE.search(i.filename))
    print(f"[preproc] total samples: {total}")

    with h5py.File(args.out, "w") as out_f:
        Q_ds = out_f.create_dataset(
            "Q", (total,) + ds_shape, dtype=out_dtype,
            compression="lzf", chunks=(1,) + ds_shape,
        )
        mask_ds = out_f.create_dataset(
            "mask", (total,) + ds_shape, dtype="uint8",
            compression="lzf", chunks=(1,) + ds_shape,
        )
        sincos_ds = out_f.create_dataset(
            "phases_sincos", (total, 2, 256), dtype="float32",
        )
        phase_raw_ds = out_f.create_dataset(
            "phases_rad", (total, 256), dtype="float32",
        )
        sim_id_ds = out_f.create_dataset(
            "sim_id", (total,), dtype="int32",
        )
        target_ds = out_f.create_dataset(
            "target_pt_m", (total, 3), dtype="float32",
        )

        out_f.attrs["raw_shape"] = np.asarray(raw_shape, dtype=np.int32)
        out_f.attrs["ds_shape"] = np.asarray(ds_shape, dtype=np.int32)
        out_f.attrs["downsample"] = int(args.downsample)
        out_f.attrs["y_offset"] = float(args.y_offset)
        out_f.attrs["storage_dtype"] = str(args.dtype)
        out_f.attrs["source_zips"] = np.asarray(
            [zp.name for zp in zip_paths], dtype=object
        )

        t0 = time.time()
        factor = 1.0 / args.downsample
        written = 0
        for sid, buf in iter_mat_files(zip_paths):
            sample = load_mat_from_bytes(buf)
            Q = sample["Q_heat"]
            valid = (~np.isnan(Q)).astype(np.uint8)
            Q = np.nan_to_num(Q, nan=0.0, copy=False)

            # downsample
            if args.downsample != 1:
                Q_small = zoom(Q, factor, order=1).astype(np.float32)
                valid_small = (
                    zoom(valid.astype(np.float32), factor, order=0) > 0.5
                ).astype(np.uint8)
            else:
                Q_small = Q
                valid_small = valid

            # log1p normalisation applied at storage time
            Q_log = np.log1p(np.clip(Q_small, 0, None) / args.y_offset)
            Q_stored = Q_log.astype(out_dtype)

            # crop / pad to target shape (zoom can be off by 1 on some sizes)
            slc = tuple(slice(0, s) for s in ds_shape)
            Q_ds[written] = Q_stored[slc]
            mask_ds[written] = valid_small[slc]

            ph = sample["phases"]
            if ph.shape != (256,):
                ph = np.asarray(ph).reshape(-1)[:256]
            sincos_ds[written, 0] = np.sin(ph).astype(np.float32)
            sincos_ds[written, 1] = np.cos(ph).astype(np.float32)
            phase_raw_ds[written] = ph.astype(np.float32)

            sim_id_ds[written] = sid
            target_ds[written] = sample["target_pt"].astype(np.float32)

            # optional cross-check against CSV
            if sid in phase_df.index:
                csv_ph = phase_df.loc[sid, phase_cols].to_numpy(dtype=np.float64)
                diff = float(np.max(np.abs(np.angle(
                    np.exp(1j * (ph - csv_ph))
                ))))
                if diff > 1e-3:
                    print(f"    [warn] sim {sid}: phases differ from CSV "
                          f"max_abs={diff:.4f}")

            written += 1
            if written % 10 == 0 or written == total:
                elapsed = time.time() - t0
                rate = elapsed / written
                print(f"[{written:4d}/{total}]  sid={sid:4d}  "
                      f"elapsed={elapsed:5.1f}s  rate={rate:.2f}s/sample")

        # trim in case some sim_ids were missing
        if written < total:
            for name in ("Q", "mask", "phases_sincos", "phases_rad",
                         "sim_id", "target_pt_m"):
                ds = out_f[name]
                ds.resize((written,) + ds.shape[1:])

    print(f"\n[preproc] wrote {written} samples to {args.out}")
    print(f"[preproc] file size: {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
