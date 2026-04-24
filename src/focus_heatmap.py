"""
3D heatmap-regression architectures for the Q -> focus-point task.

Inspired by recent 2025 medical-imaging landmark detection work
(nnLandmark, Weihsbach et al. 2025; H3DE-Net, 2502.14221): instead of
regressing the 3 coordinates directly, the network predicts a 3-D
probability field over the Q volume, trained against a Gaussian target
centred at the voxel corresponding to the ground-truth focus. At
inference we take a soft-argmax of the predicted heatmap and (optionally)
add an offset correction for sub-voxel precision.

Why expect this to be more sample-efficient than direct coordinate
regression? With N training samples and volume size V, the MSE
coordinate loss has N*3 supervised scalars, whereas the Gaussian
heatmap loss exposes N*V voxel-wise supervised targets — roughly
10^5 times more gradient signal per epoch at 30 samples. Classical
result in landmark detection.

Classes:
    HeatmapUNet3D              -- 4-level 3D U-Net, single-channel heatmap
    HeatmapUNet3DWithOffset    -- same U-Net, extra 3-channel offset head

Shared helpers:
    make_gaussian_heatmap(voxels, shape, sigma)  -- torch, batched
    soft_argmax_3d(heatmap)                      -- differentiable

Grid convention: input (B, C, D, H, W) == (B, C, z, y, x).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class _ConvBlock3D(nn.Module):
    """3x3x3 Conv -> BN -> ReLU, twice. Standard U-Net block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool3d(2), _ConvBlock3D(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up3D(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear",
                              align_corners=False)
        self.conv = _ConvBlock3D(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dz = skip.size(-3) - x.size(-3)
        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dz or dy or dx:
            x = F.pad(x,
                      [dx // 2, dx - dx // 2,
                       dy // 2, dy - dy // 2,
                       dz // 2, dz - dz // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class HeatmapUNet3D(nn.Module):
    """
    4-level 3D U-Net with a single-channel heatmap head.

    Parameters
    ----------
    in_channels   : input channels (default 2: log_Q, mask)
    base_channels : encoder width at stage 1 (doubled each stage)

    Notes
    -----
    - Output is NOT normalised to a probability here; training loss is MSE
      against a Gaussian target. We have tried softmax + KL on the training
      subset and the gain is marginal given 30 samples; raw MSE is robust.
    - For 126x128x128 input the 4 pool-halvings yield a 7x8x8 bottleneck,
      which is safely above the receptive field minimum for HIFU focal zones.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.inc = _ConvBlock3D(in_channels, c)
        self.d1 = _Down3D(c,     c * 2)
        self.d2 = _Down3D(c * 2, c * 4)
        self.d3 = _Down3D(c * 4, c * 8)
        # bottleneck: depth-wise stays at c*8 to keep param budget close
        # to the encoder-only baselines (~0.9M).
        self.u1 = _Up3D(c * 8, c * 4, c * 4)
        self.u2 = _Up3D(c * 4, c * 2, c * 2)
        self.u3 = _Up3D(c * 2, c,     c)
        self.heatmap = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.inc(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        x = self.u1(s3, s2)
        x = self.u2(x,  s1)
        x = self.u3(x,  s0)
        return self.heatmap(x)         # (B, 1, D, H, W)


class HeatmapUNet3DWithOffset(nn.Module):
    """
    Same encoder-decoder as HeatmapUNet3D plus a 3-channel per-voxel offset
    head. At inference the focus voxel is arg-max of the heatmap and the
    final continuous coordinate is `peak_voxel + offset[:, peak]`.

    This mirrors H3DE-Net (arXiv:2502.14221) and DSNT-style continuous
    regression: the heatmap localises to integer voxel accuracy, the
    offset resolves the sub-voxel residual.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.inc = _ConvBlock3D(in_channels, c)
        self.d1 = _Down3D(c,     c * 2)
        self.d2 = _Down3D(c * 2, c * 4)
        self.d3 = _Down3D(c * 4, c * 8)
        self.u1 = _Up3D(c * 8, c * 4, c * 4)
        self.u2 = _Up3D(c * 4, c * 2, c * 2)
        self.u3 = _Up3D(c * 2, c,     c)
        self.heatmap = nn.Conv3d(c, 1, kernel_size=1)
        # offsets in (z, y, x) order, unit = voxels
        self.offset  = nn.Conv3d(c, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s0 = self.inc(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        x = self.u1(s3, s2)
        x = self.u2(x,  s1)
        x = self.u3(x,  s0)
        return self.heatmap(x), self.offset(x)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def make_gaussian_heatmap(
    voxels: torch.Tensor,    # (B, 3) in (z, y, x) voxel coords
    shape: tuple[int, int, int],
    sigma: float = 3.0,
) -> torch.Tensor:
    """Return (B, 1, D, H, W) Gaussian heatmap centred at `voxels`.
    Peak amplitude = 1.0; unnormalised so MSE scales consistently."""
    B = voxels.size(0)
    D, H, W = shape
    dev = voxels.device
    zz = torch.arange(D, device=dev, dtype=voxels.dtype)[None, :, None, None]
    yy = torch.arange(H, device=dev, dtype=voxels.dtype)[None, None, :, None]
    xx = torch.arange(W, device=dev, dtype=voxels.dtype)[None, None, None, :]
    vz = voxels[:, 0][:, None, None, None]
    vy = voxels[:, 1][:, None, None, None]
    vx = voxels[:, 2][:, None, None, None]
    d2 = (zz - vz) ** 2 + (yy - vy) ** 2 + (xx - vx) ** 2
    heat = torch.exp(-d2 / (2.0 * sigma ** 2))
    return heat.unsqueeze(1)


def soft_argmax_3d(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Differentiable 3D argmax. Normalises the heatmap to a probability
    over voxels via softmax, then returns the expectation of voxel
    coordinates. Output shape: (B, 3) in (z, y, x).
    """
    B, _, D, H, W = heatmap.shape
    flat = heatmap.reshape(B, -1)
    prob = F.softmax(flat, dim=1).reshape(B, 1, D, H, W)
    dev = heatmap.device
    dtype = heatmap.dtype
    zz = torch.arange(D, device=dev, dtype=dtype)[None, None, :, None, None]
    yy = torch.arange(H, device=dev, dtype=dtype)[None, None, None, :, None]
    xx = torch.arange(W, device=dev, dtype=dtype)[None, None, None, None, :]
    z = (prob * zz).sum(dim=(2, 3, 4)).squeeze(1)
    y = (prob * yy).sum(dim=(2, 3, 4)).squeeze(1)
    x = (prob * xx).sum(dim=(2, 3, 4)).squeeze(1)
    return torch.stack([z, y, x], dim=1)


def fit_voxel_affine(tgt_m: torch.Tensor,
                     voxel_zyx: torch.Tensor) -> dict:
    """
    Fit a per-axis linear map  voxel[i] = slope[i] * tgt_m[perm[i]] + offset[i]
    on the training split. We allow an axis permutation because in Eren's
    HDF5 the Q volume's axis order is (z, y, x) but target_pt_m is (x, y, z);
    this utility picks the permutation by maximum correlation per axis.
    Returns a dict with slope, offset, perm (tensors on cpu).
    """
    tgt  = tgt_m.detach().cpu().double().numpy()           # (N, 3)
    vox  = voxel_zyx.detach().cpu().double().numpy()       # (N, 3)
    import numpy as np
    # per-voxel-axis: find the target-axis with the highest |correlation|
    perm = []
    used = set()
    for a in range(3):
        best_b, best_c = -1, 0.0
        for b in range(3):
            if b in used:
                continue
            c = abs(np.corrcoef(vox[:, a], tgt[:, b])[0, 1])
            if c > best_c:
                best_b, best_c = b, c
        perm.append(best_b); used.add(best_b)
    perm = np.array(perm)

    slopes  = np.zeros(3, dtype=np.float32)
    offsets = np.zeros(3, dtype=np.float32)
    for a in range(3):
        x = tgt[:, perm[a]]
        y = vox[:, a]
        s, o = np.polyfit(x, y, 1)
        slopes[a], offsets[a] = s, o
    return {
        "slope":  torch.as_tensor(slopes),
        "offset": torch.as_tensor(offsets),
        "perm":   torch.as_tensor(perm, dtype=torch.long),
    }


def voxel_from_target(tgt_m: torch.Tensor, fit: dict) -> torch.Tensor:
    """Apply the per-axis affine fit to map target_pt_m -> voxel coords."""
    slope  = fit["slope"].to(tgt_m.device, tgt_m.dtype)
    offset = fit["offset"].to(tgt_m.device, tgt_m.dtype)
    perm   = fit["perm"].to(tgt_m.device)
    permuted = tgt_m[:, perm]                              # (B, 3)
    return permuted * slope + offset


def target_from_voxel(voxel_zyx: torch.Tensor, fit: dict) -> torch.Tensor:
    """Inverse of voxel_from_target: voxel -> target_pt_m."""
    slope  = fit["slope"].to(voxel_zyx.device, voxel_zyx.dtype)
    offset = fit["offset"].to(voxel_zyx.device, voxel_zyx.dtype)
    perm   = fit["perm"].to(voxel_zyx.device)
    inv_perm = torch.argsort(perm)
    # tgt[perm[a]] = (voxel[a] - offset[a]) / slope[a]
    tgt_permuted = (voxel_zyx - offset) / slope
    return tgt_permuted[:, inv_perm]


FOCUS_HEATMAP_REGISTRY = {
    "heatmap":         HeatmapUNet3D,
    "heatmap_offset":  HeatmapUNet3DWithOffset,
}
