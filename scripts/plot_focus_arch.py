"""
Draw the FocusPointNet architecture as a clean flowchart PNG.

Output: outputs/focus_point/architecture.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "focus_point" / "architecture.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, title, subtitle, facecolor, edgecolor="#333"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            linewidth=1.2, facecolor=facecolor, edgecolor=edgecolor,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top",
                fontsize=10.5, fontweight="bold", color="#111")
        ax.text(x + w / 2, y + h - 0.78, subtitle, ha="center", va="top",
                fontsize=8.5, color="#333", family="monospace")

    def arrow(x1, y1, x2, y2):
        a = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="->,head_length=6,head_width=5",
            linewidth=1.3, color="#333", mutation_scale=12,
        )
        ax.add_patch(a)

    # color palette
    c_in  = "#e8f0fc"
    c_blk = "#d9ead3"
    c_pool = "#fff2cc"
    c_mlp  = "#f4cccc"
    c_out  = "#cfe2f3"

    # input
    box(0.1, 2.2, 1.8, 1.6,
        "Input\n(Q, mask)",
        "(2, 126, 128, 128)\nlog1p + standardise", c_in)

    arrow(1.9, 3.0, 2.4, 3.0)

    # 4 conv blocks - each: Conv3d x2 + BN + ReLU, then MaxPool3d(2)
    stages = [
        ("Conv3dBlock",    "16 ch\n126³",        "MaxPool3d ×2", "63³"),
        ("Conv3dBlock",    "32 ch\n63³",         "MaxPool3d ×2", "31³"),
        ("Conv3dBlock",    "64 ch\n31³",         "MaxPool3d ×2", "15³"),
        ("Conv3dBlock",    "128 ch\n15³",        "GlobalAvgPool", "(128,)"),
    ]
    x = 2.4
    for i, (t, s, pt, ps) in enumerate(stages):
        box(x, 3.25, 1.55, 1.2, t, s, c_blk)
        pool_color = c_pool if i < 3 else c_mlp
        box(x, 1.85, 1.55, 1.15, pt, ps, pool_color)
        # arrow inside column: conv -> pool
        arrow(x + 0.775, 3.25, x + 0.775, 3.0)
        # inter-column arrow
        if i < 3:
            arrow(x + 1.55, 2.42, x + 1.72, 2.42)
        x += 1.72

    # MLP head
    box(x + 0.1, 2.2, 1.6, 1.6,
        "MLP Head",
        "Linear 128→64\nGELU + Dropout\nLinear 64→3", c_mlp)
    arrow(x, 2.42, x + 0.1, 2.42)

    # Output
    box(x + 1.9, 2.2, 1.7, 1.6,
        "Focus Point",
        "(x, y, z)\nstandardised → mm", c_out)
    arrow(x + 1.7, 3.0, x + 1.9, 3.0)

    # annotations
    ax.text(7.0, 5.4, "FocusPointNet – 3D CNN Regressor (0.89 M params)",
            ha="center", fontsize=13.5, fontweight="bold")
    ax.text(7.0, 5.05,
            "Q heatmap volume (3D)  →  focus coordinate (3 DOF)",
            ha="center", fontsize=10.5, color="#333")

    # legend bar
    ax.text(0.1, 0.8, "Conv3dBlock = 3×3×3 Conv → BN → ReLU → 3×3×3 Conv → BN → ReLU",
            fontsize=8.5, color="#333")
    ax.text(0.1, 0.45,
            "Training: 30 sample (22/4/4 split), LR 3e-4, AdamW + Cosine, grad-clip 1.0, 300 epoch",
            fontsize=8.5, color="#333")
    ax.text(0.1, 0.1,
            "Downstream: predicted focus → analytical beamforming  →  256 transducer phases",
            fontsize=8.5, color="#444", style="italic")

    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"[arch] wrote {OUT}")


if __name__ == "__main__":
    main()
