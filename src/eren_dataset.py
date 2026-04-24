"""
Dataset wrapper for Eren's preprocessed HIFU dataset (data/eren/dataset.h5).

Layout (see scripts/preprocess_eren.py):
    Q              : (N, D, H, W) float16  -- log1p(Q / y_offset)
    mask           : (N, D, H, W) uint8    -- valid voxels (simulation domain)
    phases_sincos  : (N, 2, 256)  float32  -- (sin, cos) encoding
    phases_rad     : (N, 256)     float32
    sim_id         : (N,) int32
    target_pt_m    : (N, 3) float32
    attrs          : downsample, y_offset, raw_shape, ds_shape, ...

We keep things simple for the inverse problem:
    input  (x) : (2, D, H, W)  -- (log_Q, mask) both float32
    target (y) : (2, 256)       -- (sin, cos)

Q is further normalised by dividing by the training-split max so the model
input lives roughly in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ErenPhaseDataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        indices: np.ndarray | None = None,
        stats: dict | None = None,
        cache_in_ram: bool = True,
    ) -> None:
        self.h5_path = Path(h5_path)
        self.cache_in_ram = cache_in_ram

        with h5py.File(self.h5_path, "r") as f:
            n = f["Q"].shape[0]
            self.ds_shape = tuple(int(v) for v in f.attrs["ds_shape"])
        self.indices = np.arange(n) if indices is None else np.asarray(indices)
        self.stats = stats or self._compute_stats()

        self._x_cache = None
        self._y_cache = None
        if cache_in_ram:
            self._load_cache()

    # ------------------------------------------------------------------ #
    def _compute_stats(self) -> dict:
        sort_idx = np.sort(self.indices)
        with h5py.File(self.h5_path, "r") as f:
            q = f["Q"][sort_idx].astype(np.float32)
            t = (f["target_pt_m"][sort_idx].astype(np.float32)
                 if "target_pt_m" in f else None)
        q_max = float(q.max() + 1e-8)
        stats = {"q_max": q_max}
        if t is not None:
            stats["tgt_mean"] = t.mean(axis=0).astype(np.float32)
            stats["tgt_std"]  = (t.std(axis=0) + 1e-6).astype(np.float32)
        else:
            stats["tgt_mean"] = np.zeros(3, dtype=np.float32)
            stats["tgt_std"]  = np.ones(3,  dtype=np.float32)
        return stats

    # ------------------------------------------------------------------ #
    def _has_mask(self) -> bool:
        with h5py.File(self.h5_path, "r") as f:
            return "mask" in f

    # ------------------------------------------------------------------ #
    def _load_cache(self) -> None:
        sort_idx = np.sort(self.indices)
        inv = np.argsort(np.argsort(self.indices))
        has_mask = self._has_mask()
        with h5py.File(self.h5_path, "r") as f:
            q = f["Q"][sort_idx].astype(np.float32)
            m = f["mask"][sort_idx].astype(np.float32) if has_mask else None
            y = f["phases_sincos"][sort_idx].astype(np.float32)
            t = f["target_pt_m"][sort_idx].astype(np.float32) \
                if "target_pt_m" in f else None
        q = q / self.stats["q_max"]           # -> [0, 1]
        if has_mask:
            x = np.stack([q, m], axis=1)      # (n, 2, D, H, W)
        else:
            x = np.stack([q, np.ones_like(q)], axis=1)
        self._x_cache = x[inv].astype(np.float32)
        self._y_cache = y[inv].astype(np.float32)
        if t is not None:
            t_norm = (t - self.stats["tgt_mean"]) / self.stats["tgt_std"]
            self._t_cache = t_norm[inv].astype(np.float32)
        else:
            self._t_cache = None

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int) -> dict:
        if self._x_cache is not None:
            item = {
                "x": torch.from_numpy(self._x_cache[i]),
                "y": torch.from_numpy(self._y_cache[i]),
            }
            if self._t_cache is not None:
                item["target_pt"] = torch.from_numpy(self._t_cache[i])
            return item
        idx = int(self.indices[i])
        has_mask = self._has_mask()
        with h5py.File(self.h5_path, "r") as f:
            q = f["Q"][idx].astype(np.float32) / self.stats["q_max"]
            m = (f["mask"][idx].astype(np.float32) if has_mask
                 else np.ones_like(q))
            y = f["phases_sincos"][idx].astype(np.float32)
            t = (f["target_pt_m"][idx].astype(np.float32)
                 if "target_pt_m" in f else None)
        x = np.stack([q, m], axis=0)
        item = {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}
        if t is not None:
            t_norm = (t - self.stats["tgt_mean"]) / self.stats["tgt_std"]
            item["target_pt"] = torch.from_numpy(t_norm.astype(np.float32))
        return item
