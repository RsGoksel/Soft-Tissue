"""
2D ConvNeXt-style encoder/decoder for pressure-field prediction.

Keeps the same I/O contract as UNet2d (in: c, rho, alpha; out: peak
pressure) so scripts/train_convnext.py can reuse train_unet.py's
dataset, loss, and splits for a clean head-to-head.

Blocks:
  ConvNeXtBlock = DWConv(7x7) -> LayerNorm -> 1x1(4c) -> GELU -> 1x1(c) + skip
  Down          = Conv2d stride-2 (learned downsampling)
  Up            = bilinear upsample + 1x1 conv to fuse skip

Design choice: stride-1 depthwise conv (spatial only) in each stage, and
learned stride-2 projections between stages. This matches ConvNeXt-V1
(Liu et al. 2022) and keeps the parameter budget comparable with UNet2d
at base_channels=32.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """channels-first LayerNorm (keeps (N, C, H, W) layout)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 4,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm   = LayerNorm2d(dim)
        self.pw1    = nn.Conv2d(dim, dim * expand, kernel_size=1)
        self.act    = nn.GELU()
        self.pw2    = nn.Conv2d(dim * expand, dim, kernel_size=1)
        self.drop_path = nn.Identity() if drop_path <= 0 else nn.Dropout2d(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return residual + self.drop_path(x)


class _Stage(nn.Module):
    def __init__(self, dim: int, n_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ConvNeXtBlock(dim) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvNeXt2d(nn.Module):
    """
    ConvNeXt encoder + lightweight decoder with U-Net-style skips.

    Parameters
    ----------
    in_channels   : input channels (default 3 for c, rho, alpha)
    out_channels  : output channels (default 1 for peak pressure)
    base_channels : width of the first stage (doubled at each downsample)
    depths        : number of ConvNeXt blocks per stage (4 stages)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depths: tuple[int, int, int, int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2, c3, c4 = c1 * 2, c1 * 4, c1 * 8

        # stem: 1x1 conv to embed to c1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            LayerNorm2d(c1),
        )
        self.s1 = _Stage(c1, depths[0])
        self.d1 = nn.Sequential(LayerNorm2d(c1),
                                nn.Conv2d(c1, c2, kernel_size=2, stride=2))
        self.s2 = _Stage(c2, depths[1])
        self.d2 = nn.Sequential(LayerNorm2d(c2),
                                nn.Conv2d(c2, c3, kernel_size=2, stride=2))
        self.s3 = _Stage(c3, depths[2])
        self.d3 = nn.Sequential(LayerNorm2d(c3),
                                nn.Conv2d(c3, c4, kernel_size=2, stride=2))
        self.s4 = _Stage(c4, depths[3])

        # decoder: simple bilinear up + 1x1 fuse + one ConvNeXt block each.
        self.u3 = nn.Sequential(nn.Conv2d(c4 + c3, c3, kernel_size=1),
                                ConvNeXtBlock(c3))
        self.u2 = nn.Sequential(nn.Conv2d(c3 + c2, c2, kernel_size=1),
                                ConvNeXtBlock(c2))
        self.u1 = nn.Sequential(nn.Conv2d(c2 + c1, c1, kernel_size=1),
                                ConvNeXtBlock(c1))

        self.head = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _up(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad so each downsample halves cleanly
        H0, W0 = x.shape[-2:]
        pad_h = (8 - H0 % 8) % 8
        pad_w = (8 - W0 % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

        s1 = self.s1(self.stem(x))
        s2 = self.s2(self.d1(s1))
        s3 = self.s3(self.d2(s2))
        s4 = self.s4(self.d3(s3))

        x = self.u3(self._up(s4, s3))
        x = self.u2(self._up(x,  s2))
        x = self.u1(self._up(x,  s1))
        x = self.head(x)
        return x[..., :H0, :W0]
