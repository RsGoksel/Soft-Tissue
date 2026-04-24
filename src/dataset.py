"""
PyTorch Dataset wrapper around the HDF5 file produced by
scripts/generate_dataset.py. Handles per-channel normalization so the FNO
sees zero-mean / unit-variance inputs and a scaled output.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PressureFieldDataset(Dataset):
    """
    HDF5 layout:
        inputs  : (N, 3, H, W)  float32  -- (c, rho, alpha)
        targets : (N, 1, H, W)  float32  -- peak pressure [Pa]
        focus   : (N, 2)        int32    -- target (y, x)

    Target normalization
    --------------------
    Peak pressure spans 4+ orders of magnitude per sample (kPa near the
    sensor edges, MPa at the focus). A linear scale pushes the network to
    ignore low-pressure background; a log transform gives each pixel
    similar weight. We apply:

        y' = log1p(p_max / y_offset)
        y' /= log_scale   # so training targets live in ~[0, 1]

    Inference denormalization:

        p_max = (exp(y' * log_scale) - 1) * y_offset
    """

    def __init__(
        self,
        h5_path: str | Path,
        indices: np.ndarray | None = None,
        stats: dict | None = None,
        log_target: bool = True,
        y_offset: float = 1e4,  # 10 kPa floor
        cache_in_ram: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        self._f: h5py.File | None = None
        self.log_target = log_target
        self.y_offset = float(y_offset)
        self.cache_in_ram = cache_in_ram

        with h5py.File(self.h5_path, "r") as f:
            n = f["inputs"].shape[0]
        self.indices = np.arange(n) if indices is None else np.asarray(indices)

        self.stats = stats or self._compute_stats()

        # pre-load the whole split into RAM so the training loop never
        # touches disk (per-sample HDF5 reads are the main bottleneck)
        self._x_cache: np.ndarray | None = None
        self._y_cache: np.ndarray | None = None
        if self.cache_in_ram:
            self._load_cache()

    def _load_cache(self) -> None:
        sort_idx = np.sort(self.indices)
        inv = np.argsort(np.argsort(self.indices))  # map sorted -> requested
        with h5py.File(self.h5_path, "r") as f:
            x_all = f["inputs"][sort_idx]   # (n, 3, H, W)
            y_all = f["targets"][sort_idx]  # (n, 1, H, W)
        # normalise once, up-front
        x_all = (x_all - self.stats["x_mean"]) / self.stats["x_std"]
        if self.stats.get("log_target", False):
            y_all = np.log1p(np.clip(y_all, 0, None) / self.stats["y_offset"])
            y_all = y_all / self.stats["log_scale"]
        else:
            y_all = y_all / self.stats["y_scale"]
        # restore original (requested) ordering
        self._x_cache = x_all[inv].astype(np.float32)
        self._y_cache = y_all[inv].astype(np.float32)

    # ------------------------------------------------------------------ #
    def _compute_stats(self) -> dict:
        """Per-channel mean/std for inputs; log-or-max scaling for output."""
        sort_idx = np.sort(self.indices)
        with h5py.File(self.h5_path, "r") as f:
            x = f["inputs"][sort_idx]    # (n, 3, H, W)
            y = f["targets"][sort_idx]   # (n, 1, H, W)

        stats = {
            "x_mean": x.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32),
            "x_std":  (x.std(axis=(0, 2, 3), keepdims=True) + 1e-8).astype(np.float32),
            "log_target": self.log_target,
            "y_offset": self.y_offset,
        }
        if self.log_target:
            y_log = np.log1p(np.clip(y, 0, None) / self.y_offset)
            stats["log_scale"] = float(y_log.max() + 1e-8)
            stats["y_scale"] = None
        else:
            stats["y_scale"] = float(np.abs(y).max() + 1e-8)
            stats["log_scale"] = None
        return stats

    # ------------------------------------------------------------------ #
    def _lazy_open(self) -> h5py.File:
        if self._f is None:
            # swmr=False is fine since we only read after generation is done
            self._f = h5py.File(self.h5_path, "r")
        return self._f

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int) -> dict:
        if self._x_cache is not None:
            return {
                "x": torch.from_numpy(self._x_cache[i]),
                "y": torch.from_numpy(self._y_cache[i]),
            }
        idx = int(self.indices[i])
        f = self._lazy_open()
        x = f["inputs"][idx]      # (3, H, W)
        y = f["targets"][idx]     # (1, H, W)

        x = (x - self.stats["x_mean"][0]) / self.stats["x_std"][0]
        if self.stats.get("log_target", False):
            y = np.log1p(np.clip(y, 0, None) / self.stats["y_offset"])
            y = y / self.stats["log_scale"]
        else:
            y = y / self.stats["y_scale"]

        return {
            "x": torch.from_numpy(x.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.float32)),
        }

    def denormalize_target(self, y_norm: np.ndarray) -> np.ndarray:
        """Invert the target normalization (useful for plotting predictions
        in physical units)."""
        if self.stats.get("log_target", False):
            y = y_norm * self.stats["log_scale"]
            return (np.expm1(y)) * self.stats["y_offset"]
        return y_norm * self.stats["y_scale"]

    def __del__(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
