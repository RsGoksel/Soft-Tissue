"""
Phantom loader + augmentation.

Two sources:
    1. load_oa_breast_slice(...)   -- real OA-Breast phantom, 2D axial slice
    2. make_synthetic_phantom(...) -- quick procedural phantom for pipeline tests

Both return an integer label map compatible with tissue_properties.py.
insert_tumor(...) randomly places a tumor (label=TUMOR) inside tissue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from tissue_properties import (
    BACKGROUND, FAT, FIBROGLANDULAR, SKIN, TUMOR, VESSEL,
)

# OpenBreastUS tissue classes (inferred from the 4 phantom types the dataset
# advertises: heterogeneous / fibroglandular / fatty / extremely dense). The
# dataset ships continuous sound-speed maps rather than discrete labels, so we
# quantise them back into our label space using the reference speeds defined
# in tissue_properties.py. See load_openbreastus() below.


# --------------------------------------------------------------------------- #
# Synthetic phantom (used when OA-Breast is not yet downloaded)
# --------------------------------------------------------------------------- #
def make_synthetic_phantom(
    nx: int = 256,
    ny: int = 256,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build a crude 2D axial breast phantom:
      - skin (thin shell, label=4)
      - fat body (ellipse, label=3)
      - fibroglandular blobs inside (label=2)
      - background = water (label=0)
    """
    rng = rng or np.random.default_rng()
    img = np.zeros((nx, ny), dtype=np.uint8)

    cy, cx = nx // 2, ny // 2
    yy, xx = np.mgrid[0:nx, 0:ny]

    # fat ellipse (the breast body)
    a = rng.integers(int(nx * 0.30), int(nx * 0.42))
    b = rng.integers(int(ny * 0.25), int(ny * 0.38))
    fat_mask = ((yy - cy) / a) ** 2 + ((xx - cx) / b) ** 2 <= 1.0
    img[fat_mask] = FAT

    # skin: 2-pixel shell around fat
    from scipy.ndimage import binary_dilation
    skin_mask = binary_dilation(fat_mask, iterations=2) & ~fat_mask
    img[skin_mask] = SKIN

    # a handful of fibroglandular blobs inside the fat region
    n_blobs = rng.integers(3, 8)
    for _ in range(n_blobs):
        by = rng.integers(cy - a // 2, cy + a // 2)
        bx = rng.integers(cx - b // 2, cx + b // 2)
        br = rng.integers(5, 18)
        blob = (yy - by) ** 2 + (xx - bx) ** 2 <= br ** 2
        img[blob & fat_mask] = FIBROGLANDULAR

    return img


# --------------------------------------------------------------------------- #
# OA-Breast loader (real phantoms)
# --------------------------------------------------------------------------- #
def load_oa_breast_slice(
    dat_path: str | Path,
    shape: tuple[int, int, int],
    axial_index: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Load one 2D axial slice from an OA-Breast MergedPhantom.DAT file.

    Parameters
    ----------
    dat_path : path to MergedPhantom.DAT (uint8 binary)
    shape    : (Nz, Ny, Nx) of the stored volume. OA-Breast does not embed
               dimensions in the .DAT file; read them from the companion README
               or the HDF5 mirror. Common sizes are listed in the README.md.
    axial_index : which axial slice to return (z index). None -> random.

    Returns
    -------
    label_map : uint8 array of shape (Ny, Nx) with OA-Breast labels.
    """
    rng = rng or np.random.default_rng()
    vol = np.fromfile(dat_path, dtype=np.uint8).reshape(shape)

    if axial_index is None:
        # avoid empty slices at the very top/bottom
        lo = int(0.2 * shape[0])
        hi = int(0.8 * shape[0])
        axial_index = int(rng.integers(lo, hi))

    return vol[axial_index]


# --------------------------------------------------------------------------- #
# OpenBreastUS loader (Zeng et al. 2025, arXiv:2507.15035)
# https://huggingface.co/datasets/OpenBreastUS/breast
# --------------------------------------------------------------------------- #
def load_openbreastus_speedmaps(
    mat_path: str | Path,
    key: str | None = None,
) -> np.ndarray:
    """
    Load the raw sound-speed stack stored in one of the OpenBreastUS
    `breast_{train,test}_speed.mat` files.

    Handles BOTH MATLAB v7 (scipy.io.loadmat, test file, leading index) and
    MATLAB v7.3 / HDF5 (h5py, train file, trailing index -- column-major).

    Verified layouts:
        test (v7, 195 MB):  'breast_test',  (800, 480, 480)  leading idx
        train (v7.3, 1.69 GB): 'breast_train', (480, 480, 7200) trailing idx

    This function always returns an array with the PHANTOM INDEX FIRST:
        (N, 480, 480), dtype float32, sound speed in m/s.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(
            f"{mat_path} not found. Download from "
            f"https://huggingface.co/datasets/OpenBreastUS/breast"
        )

    # detect format from file header
    with open(mat_path, "rb") as f:
        header = f.read(16)
    is_v73 = header.startswith(b"MATLAB 7.3")

    candidate_keys = [key] if key else ["breast_test", "breast_train"]

    if is_v73:
        import h5py
        with h5py.File(mat_path, "r") as f:
            present = [k for k in f.keys() if not k.startswith("#")]
            for k in candidate_keys:
                if k and k in f:
                    arr = f[k][...]  # materialise
                    break
            else:
                raise KeyError(
                    f"No known OpenBreastUS key in {mat_path}. "
                    f"Present: {present}. Retry with key='<name>'."
                )
        arr = np.asarray(arr).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError(f"expected 3D array, got shape {arr.shape}")
        # h5py returns the raw HDF5 layout -- for this dataset the shape is
        # (H, W, N). Move the phantom axis to the front.
        axes = sorted(range(3), key=lambda i: arr.shape[i])
        # largest axis = phantom index, put it first; keep the other two
        # in their original order so 2D slices remain oriented correctly.
        n_axis = int(np.argmax(arr.shape))
        other = [i for i in range(3) if i != n_axis]
        arr = np.transpose(arr, (n_axis, *other))
        return arr

    # legacy v7 / v5 path: use scipy.io.loadmat
    from scipy.io import loadmat
    raw = loadmat(mat_path)
    for k in candidate_keys:
        if k and k in raw:
            arr = np.asarray(raw[k]).astype(np.float32)
            if arr.ndim == 3:
                if arr.shape[0] < arr.shape[-1]:
                    return arr
                return np.transpose(arr, (2, 0, 1))
            arr = np.squeeze(arr)
            if arr.ndim == 3:
                return arr
    real_keys = [k for k in raw.keys() if not k.startswith("__")]
    raise KeyError(
        f"No known OpenBreastUS array key found. Keys present: {real_keys}. "
        f"Retry with load_openbreastus_speedmaps(path, key='<name>')."
    )


def openbreastus_to_label_map(
    speed_map: np.ndarray,
    bg_thresh: float = 1420.0,
    fat_thresh: float = 1480.0,
    fibro_thresh: float = 1560.0,
) -> np.ndarray:
    """
    Quantise a continuous OpenBreastUS sound-speed slice into our discrete
    label convention so the rest of the pipeline (tumor insertion,
    tissue_properties.labels_to_maps) can consume it unchanged.

    Thresholds calibrated to the OpenBreastUS test file value distribution
    (range ~1403..1597 m/s, mean ~1491):
        c < 1420  -> background (coupling water with small numerical offset)
        1420-1480 -> fat
        1480-1560 -> fibroglandular
        > 1560    -> skin   (outer rim, highest speeds)
    """
    if speed_map.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {speed_map.shape}")

    labels = np.full(speed_map.shape, BACKGROUND, dtype=np.uint8)
    labels[(speed_map >= bg_thresh) & (speed_map < fat_thresh)] = FAT
    labels[(speed_map >= fat_thresh) & (speed_map < fibro_thresh)] = FIBROGLANDULAR
    labels[speed_map >= fibro_thresh] = SKIN
    return labels


def load_openbreastus_phantom(
    mat_path: str | Path,
    phantom_index: int,
    target_size: Optional[int] = None,
    return_speed: bool = False,
) -> np.ndarray:
    """
    Load one OpenBreastUS phantom (already 2D, 480x480) and return it as a
    label map ready for `insert_tumor(...)` and `run_focused_sim(...)`.

    Parameters
    ----------
    mat_path      : path to breast_test_speed.mat or breast_train_speed.mat
    phantom_index : int in [0, N-1]
    target_size   : optional side length to bilinearly resize to (for faster
                    simulations; pass e.g. 256). None keeps native 480.
    return_speed  : if True, also return the continuous sound-speed slice.
    """
    vol = load_openbreastus_speedmaps(mat_path)
    if phantom_index >= vol.shape[0]:
        raise IndexError(
            f"phantom_index {phantom_index} >= dataset size {vol.shape[0]}"
        )

    speed = vol[phantom_index]  # shape (480, 480), float32 m/s
    if target_size is not None and target_size != speed.shape[0]:
        from scipy.ndimage import zoom
        factor = target_size / speed.shape[0]
        speed = zoom(speed, zoom=factor, order=1).astype(np.float32)

    labels = openbreastus_to_label_map(speed)
    if return_speed:
        return labels, speed
    return labels


# kept for backward compatibility; the new entry point is
# `load_openbreastus_phantom`
load_openbreastus_slice = load_openbreastus_phantom


def load_openbreastus_speed(
    mat_path: str | Path,
    phantom_index: int,
    target_size: Optional[int] = None,
) -> np.ndarray:
    """
    NATIVE-path loader: return ONLY the continuous sound-speed map for one
    phantom, with no quantisation. Feed this directly into
    `simulate.run_focused_sim_from_speed` (after inserting a tumor via
    `insert_tumor_speed`).
    """
    vol = load_openbreastus_speedmaps(mat_path)
    if phantom_index >= vol.shape[0]:
        raise IndexError(
            f"phantom_index {phantom_index} >= dataset size {vol.shape[0]}"
        )
    speed = vol[phantom_index].astype(np.float32)
    if target_size is not None and target_size != speed.shape[0]:
        from scipy.ndimage import zoom
        factor = target_size / speed.shape[0]
        speed = zoom(speed, zoom=factor, order=1).astype(np.float32)
    return speed


# --------------------------------------------------------------------------- #
# Tumor insertion on continuous sound-speed maps (NATIVE path)
# --------------------------------------------------------------------------- #
def insert_tumor_speed(
    speed_map: np.ndarray,
    radius_range: tuple[int, int] = (8, 18),
    tumor_speed: float = 1550.0,
    host_speed_range: tuple[float, float] = (1430.0, 1560.0),
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Place one spherical tumor inside a region of a continuous sound-speed
    map (e.g. an OpenBreastUS phantom). The tumor is realised by overwriting
    voxels with `tumor_speed` (m/s) -- the Table 8 reference value.

    A host voxel is any voxel whose current speed lies in `host_speed_range`
    (so we only replace actual breast tissue, not water or skin).

    Returns
    -------
    new_speed : float32 array, copy of input with tumor baked in
    (axial, lateral, radius)
    """
    rng = rng or np.random.default_rng()
    out = speed_map.astype(np.float32, copy=True)

    lo, hi = host_speed_range
    host_mask = (speed_map >= lo) & (speed_map <= hi)
    ys, xs = np.where(host_mask)
    if ys.size == 0:
        raise ValueError(
            f"No host tissue in speed range {host_speed_range} m/s "
            f"(phantom range {speed_map.min():.1f}..{speed_map.max():.1f})."
        )

    radius = int(rng.integers(radius_range[0], radius_range[1] + 1))

    for _ in range(200):
        idx = int(rng.integers(0, ys.size))
        cy, cx = int(ys[idx]), int(xs[idx])
        if (
            radius <= cy < speed_map.shape[0] - radius
            and radius <= cx < speed_map.shape[1] - radius
        ):
            # require the whole disk to be inside host tissue, not at edge
            yy, xx = np.mgrid[
                cy - radius : cy + radius + 1,
                cx - radius : cx + radius + 1,
            ]
            disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            sub_host = host_mask[
                cy - radius : cy + radius + 1,
                cx - radius : cx + radius + 1,
            ]
            if (disk & sub_host).sum() > 0.7 * disk.sum():
                break
    else:
        raise RuntimeError("Could not place tumor inside host tissue.")

    yy, xx = np.mgrid[: speed_map.shape[0], : speed_map.shape[1]]
    tumor_mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2) & host_mask
    out[tumor_mask] = tumor_speed

    return out, (cy, cx, radius)


# --------------------------------------------------------------------------- #
# Tumor insertion on discrete label maps (LABEL path)
# --------------------------------------------------------------------------- #
def insert_tumor(
    label_map: np.ndarray,
    radius_range: tuple[int, int] = (5, 15),
    host_labels: tuple[int, ...] = (FAT, FIBROGLANDULAR),
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Place one spherical tumor randomly inside a host tissue region.

    Returns
    -------
    new_label_map : uint8 array, copy of input with tumor baked in
    (cy, cx, radius)
    """
    rng = rng or np.random.default_rng()
    out = label_map.copy()

    host_mask = np.isin(label_map, host_labels)
    ys, xs = np.where(host_mask)
    if ys.size == 0:
        raise ValueError("No host tissue found for tumor insertion.")

    radius = int(rng.integers(radius_range[0], radius_range[1] + 1))

    # reject seed points too close to the border so the sphere fits
    for _ in range(100):
        idx = int(rng.integers(0, ys.size))
        cy, cx = int(ys[idx]), int(xs[idx])
        if (
            radius <= cy < label_map.shape[0] - radius
            and radius <= cx < label_map.shape[1] - radius
        ):
            break
    else:
        raise RuntimeError("Could not place tumor inside host tissue.")

    yy, xx = np.mgrid[: label_map.shape[0], : label_map.shape[1]]
    tumor_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    # only overwrite host tissue, not skin/vessel/background
    tumor_mask &= host_mask
    out[tumor_mask] = TUMOR

    return out, (cy, cx, radius)
