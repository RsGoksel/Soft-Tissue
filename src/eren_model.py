"""
Inverse 3D CNN for Eren's HIFU phase planning problem.

Task:
    Input  : target heat deposition map Q_target(x, y, z) in [0, ~12] (log1p space)
             + validity mask (1 channel)
    Output : 256 transducer phases, encoded as sin/cos => (2, 256) tensor

Architecture:
    4-level 3D encoder (32 -> 64 -> 128 -> 256 channels, MaxPool3d between)
    -> global average pool -> (256,)
    -> MLP head: 256 -> 512 -> 512 -> 512 (= 2 * 256 outputs)
    -> reshape to (2, 256), tanh to keep within sin/cos range

Loss options:
    MSE on sin/cos  -- direct regression (simple, biased against wrap-around)
    Circular loss   -- 1 - cos(phi_pred - phi_true), handles angle wrap natively

We implement both as helper classes. Default training uses MSE on sin/cos.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhaseInverseNet(nn.Module):
    """
    3D CNN encoder + MLP head, with optional auxiliary target-point input.

    Output is 256 unit vectors (sin, cos) per transducer. We DO NOT use
    `tanh` on the final layer because it creates a trivial "predict 0"
    shortcut (all-zero output gives MSE loss = 0.5 per element, exactly the
    plateau we saw in previous runs). Instead we L2-normalise each
    (sin, cos) pair onto the unit circle -- this keeps the geometry right
    without offering a zero-gradient solution.

    Parameters
    ----------
    in_channels  : volumetric input channels (default 2: log_Q, validity_mask)
    n_transducers: number of transducer elements (default 256)
    base_channels: width of the first encoder stage
    use_target_pt: concatenate (3,) target coordinate to the MLP bottleneck
    """

    def __init__(
        self,
        in_channels: int = 2,
        n_transducers: int = 256,
        base_channels: int = 32,
        use_target_pt: bool = False,
    ) -> None:
        super().__init__()
        c = base_channels
        self.n_trans = n_transducers
        self.use_target_pt = use_target_pt

        self.stage1 = Conv3dBlock(in_channels, c)
        self.pool1  = nn.MaxPool3d(2)
        self.stage2 = Conv3dBlock(c,     c * 2)
        self.pool2  = nn.MaxPool3d(2)
        self.stage3 = Conv3dBlock(c * 2, c * 4)
        self.pool3  = nn.MaxPool3d(2)
        self.stage4 = Conv3dBlock(c * 4, c * 8)
        self.gap    = nn.AdaptiveAvgPool3d(1)

        head_in = c * 8 + (3 if use_target_pt else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2 * n_transducers),
            # no tanh! see class docstring. explicit unit-circle projection
            # happens in forward() below.
        )

    def forward(
        self,
        x: torch.Tensor,
        target_pt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x         : (B, in_channels, D, H, W)
        target_pt : (B, 3) target coordinate, required iff use_target_pt=True
        returns   : (B, 2, n_transducers)  -- channel 0 = sin, channel 1 = cos,
                    each (sin, cos) pair L2-normalised to the unit circle.
        """
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        x = self.pool3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)

        if self.use_target_pt:
            if target_pt is None:
                raise ValueError(
                    "PhaseInverseNet constructed with use_target_pt=True "
                    "but no target_pt was passed to forward()."
                )
            x = torch.cat([x, target_pt], dim=1)

        x = self.head(x)
        sc = x.reshape(-1, 2, self.n_trans)
        # project each 2-vector onto the unit circle
        norm = torch.linalg.vector_norm(sc, dim=1, keepdim=True).clamp_min(1e-6)
        return sc / norm


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class SinCosMSELoss(nn.Module):
    """Plain MSE on the (sin, cos) 2-channel output."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


class CircularPhaseLoss(nn.Module):
    """
    Wrap-aware loss: 1 - cos(phi_pred - phi_true).
    Expects sin/cos encoded inputs of shape (B, 2, N).
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # cos(a - b) = cos_a*cos_b + sin_a*sin_b
        cos_diff = pred[:, 0] * target[:, 0] + pred[:, 1] * target[:, 1]
        return (1.0 - cos_diff).mean()


class GaugeInvariantPhaseLoss(nn.Module):
    """
    Phase-pattern loss invariant to a per-sample global phase offset.
    HIFU |pressure|^2 (=> Q) is invariant under phi_i -> phi_i + c, so the
    target phase vector is only defined up to this gauge. We analytically
    remove the best-fit constant per sample before comparing.

    For each batch element we compute the complex inner product
        z = sum_i exp(i*(phi_pred_i - phi_true_i))
    and minimise 1 - |z|/N, which is 0 iff pred = true + c for some c.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (B, 2, N) with channel 0 = sin, 1 = cos
        # complex inner product <pred, conj(target)> / N
        cos_diff = pred[:, 1] * target[:, 1] + pred[:, 0] * target[:, 0]  # Re
        sin_diff = pred[:, 0] * target[:, 1] - pred[:, 1] * target[:, 0]  # Im
        re = cos_diff.mean(dim=1)
        im = sin_diff.mean(dim=1)
        mag = torch.sqrt(re * re + im * im + 1e-8)  # in [0, 1]
        return (1.0 - mag).mean()


# --------------------------------------------------------------------------- #
# Focus-point regressor (alternate prediction target)
#
# Diagnostic result (2026-04-17): the Q -> 256-phase map is not a deterministic
# function (nearest-neighbour phase similarity = 0.003, indistinguishable from
# random). However Q -> 3-coordinate focus point IS learnable even at 30
# samples (RMS 9.9 mm). For a symposium-grade preliminary result we therefore
# predict the 3 focus coordinates and let the physics-based beamformer
# compute the 256 phases analytically afterwards.
# --------------------------------------------------------------------------- #
class FocusPointNet(nn.Module):
    """
    3D CNN encoder + MLP head that predicts a STANDARDISED (mean/std per
    axis) target point from a heat-deposition volume.

    Input  : (B, in_channels=2, D, H, W)  -- (log_Q normalised, mask)
    Output : (B, 3)                        -- standardised (x, y, z); invert
                                              with dataset.stats['tgt_mean/std'].
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 16,
    ) -> None:
        super().__init__()
        c = base_channels
        self.enc = nn.Sequential(
            Conv3dBlock(in_channels, c),   nn.MaxPool3d(2),
            Conv3dBlock(c,     c * 2),     nn.MaxPool3d(2),
            Conv3dBlock(c * 2, c * 4),     nn.MaxPool3d(2),
            Conv3dBlock(c * 4, c * 8),     nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


# --------------------------------------------------------------------------- #
# Alternative focus-point architectures (2026-04-21 extension).
# Why: the symposium draft needs a backbone-robustness study. We keep the
# baseline FocusPointNet and add two competitors of comparable parameter
# budget so that any reported gain isn't a width/depth confound.
# --------------------------------------------------------------------------- #


class _ResBlock3D(nn.Module):
    """Bottleneck-style 3D residual block with BN + ReLU.
    Uses a 1x1x1 projection on the skip path whenever channels change."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.skip = (
            nn.Identity() if in_ch == out_ch
            else nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class FocusPointResNet3D(nn.Module):
    """ResNet-style 3D encoder for focus-point regression.

    Same width schedule as FocusPointNet (c, 2c, 4c, 8c) but with identity
    skip connections so deeper stacks train stably on tiny (30-sample)
    datasets. Head is unchanged so any difference in metric is attributable
    to the encoder family.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
        )
        self.stage1 = _ResBlock3D(c,     c)
        self.down1  = nn.MaxPool3d(2)
        self.stage2 = _ResBlock3D(c,     c * 2)
        self.down2  = nn.MaxPool3d(2)
        self.stage3 = _ResBlock3D(c * 2, c * 4)
        self.down3  = nn.MaxPool3d(2)
        self.stage4 = _ResBlock3D(c * 4, c * 8)
        self.gap    = nn.AdaptiveAvgPool3d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(self.stage1(x))
        x = self.down2(self.stage2(x))
        x = self.down3(self.stage3(x))
        x = self.gap(self.stage4(x))
        return self.head(x)


class FocusPointUNet3D(nn.Module):
    """3D UNet-style encoder with **multi-scale feature fusion** into the
    regression head.

    Unlike a classical UNet (which needs a decoder for dense prediction),
    we only need a 3-vector output, so the "decoder" here collapses each
    encoder stage to a global descriptor (AdaptiveAvgPool3d(1)) and
    concatenates the four scale descriptors before the MLP. This gives
    the head access to both coarse (localisation) and fine (shape)
    features without the cost of a transposed-conv decoder.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.stage1 = Conv3dBlock(in_channels, c)
        self.pool1  = nn.MaxPool3d(2)
        self.stage2 = Conv3dBlock(c,     c * 2)
        self.pool2  = nn.MaxPool3d(2)
        self.stage3 = Conv3dBlock(c * 2, c * 4)
        self.pool3  = nn.MaxPool3d(2)
        self.stage4 = Conv3dBlock(c * 4, c * 8)

        self.gap = nn.AdaptiveAvgPool3d(1)
        fused_ch = c + c * 2 + c * 4 + c * 8          # = 15c

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_ch, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stage1(x)
        s2 = self.stage2(self.pool1(s1))
        s3 = self.stage3(self.pool2(s2))
        s4 = self.stage4(self.pool3(s3))
        z = torch.cat([self.gap(s) for s in (s1, s2, s3, s4)], dim=1)
        return self.head(z)


# Central registry used by the architecture-comparison script so we keep
# the naming consistent everywhere.
FOCUS_ARCH_REGISTRY = {
    "cnn":    FocusPointNet,
    "resnet": FocusPointResNet3D,
    "unet":   FocusPointUNet3D,
}


def phase_error_degrees(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Reporting metric: mean absolute phase error in degrees, wrap-aware.
    """
    phi_pred = torch.atan2(pred[:, 0], pred[:, 1])
    phi_true = torch.atan2(target[:, 0], target[:, 1])
    diff = torch.atan2(torch.sin(phi_pred - phi_true),
                       torch.cos(phi_pred - phi_true))
    return diff.abs().mean() * (180.0 / math.pi)
