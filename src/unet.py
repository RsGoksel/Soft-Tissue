"""
Simple 2D U-Net baseline for the (c, rho, alpha) -> peak pressure regression.

Intentionally plain: 4 encoder levels, 4 decoder levels, skip connections,
BatchNorm + ReLU. No attention, no residuals, no tricks. This is the
standard "image-to-image CNN" baseline every operator-learning paper uses
as its point of comparison (TUSNet, DeepTFUS, Stanziola use variants of it).

Model maps:
    (B, in_channels=3, H, W)  ->  (B, out_channels=1, H, W)

Input H and W can be arbitrary; the network uses reflection padding and
bilinear upsampling so any multiple of 16 works. For our 316x256 padded
grid we crop the decoder output back to the input size via center crop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if shapes don't match due to odd dims after pooling
        dy = skip.size(-2) - x.size(-2)
        dx = skip.size(-1) - x.size(-1)
        if dy != 0 or dx != 0:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet2d(nn.Module):
    """
    4-level encoder/decoder U-Net.

    Parameters
    ----------
    in_channels  : number of input channels (default 3 for c, rho, alpha)
    out_channels : number of output channels (default 1 for peak pressure)
    base_channels: width of the first encoder block (doubled each level)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.d1 = Down(c,    c * 2)
        self.d2 = Down(c * 2, c * 4)
        self.d3 = Down(c * 4, c * 8)
        self.d4 = Down(c * 8, c * 16)

        self.u1 = Up(c * 16, c * 8, c * 8)
        self.u2 = Up(c * 8,  c * 4, c * 4)
        self.u3 = Up(c * 4,  c * 2, c * 2)
        self.u4 = Up(c * 2,  c,     c)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad spatial dims so they are divisible by 16 (four pool ops)
        H0, W0 = x.shape[-2:]
        pad_h = (16 - H0 % 16) % 16
        pad_w = (16 - W0 % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode="reflect")

        s0 = self.inc(x)
        s1 = self.d1(s0)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        x = self.d4(s3)

        x = self.u1(x, s3)
        x = self.u2(x, s2)
        x = self.u3(x, s1)
        x = self.u4(x, s0)
        x = self.outc(x)

        # crop back to input spatial size
        return x[..., :H0, :W0]
