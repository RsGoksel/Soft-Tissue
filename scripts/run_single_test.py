"""
End-to-end sanity check: build one synthetic phantom, insert a tumor,
run a single 2D k-wave simulation, save inputs/outputs as a .npz and a PNG
preview. This is the "1-sample pipeline works" smoke test.

Run from project root:
    python scripts/run_single_test.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phantom import insert_tumor, make_synthetic_phantom     # noqa: E402
from simulate import SimConfig, run_focused_sim              # noqa: E402


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(0)

    # 1. phantom with tumor
    labels = make_synthetic_phantom(nx=256, ny=256, rng=rng)
    labels, (cy, cx, r) = insert_tumor(labels, rng=rng)
    print(f"[phantom] tumor at (y={cy}, x={cx}), radius={r} px")

    # 2. run simulation, focusing on the tumor center
    cfg = SimConfig(use_gpu=False)  # flip to True once a CUDA GPU is available
    result = run_focused_sim(labels, focus_yx=(cy, cx), config=cfg)
    print(f"[sim] p_max range: {result['p_max'].min():.2e} .. {result['p_max'].max():.2e} Pa")

    # 3. save raw arrays
    np.savez_compressed(
        out_dir / "sample_0000.npz",
        labels=result["labels"],
        c=result["c"],
        rho=result["rho"],
        alpha=result["alpha"],
        p_max=result["p_max"],
        focus_yx=result["focus_yx"],
    )

    # 4. preview figure -- axes in mm, source line overlay, focus marker
    n_ax, n_lat = result["labels"].shape
    extent_mm = [0, n_lat * cfg.dx * 1e3, n_ax * cfg.dx * 1e3, 0]  # (L, R, B, T)

    src_ys, src_xs = np.where(result["source_mask"])

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))

    ax[0].imshow(result["labels"], cmap="tab10", extent=extent_mm)
    ax[0].set_title("Tissue labels")

    im1 = ax[1].imshow(result["c"], cmap="viridis", extent=extent_mm)
    ax[1].set_title("Sound speed [m/s]")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    # peak pressure: clip extreme values to show both focus and background
    p_max_arr = result["p_max"]
    vmax = float(np.percentile(p_max_arr, 99.5))
    im2 = ax[2].imshow(p_max_arr, cmap="hot", vmin=0, vmax=vmax, extent=extent_mm)
    # overlay source line (magenta) and focus target (cyan star)
    if src_ys.size:
        ax[2].plot(
            src_xs * cfg.dx * 1e3,
            src_ys * cfg.dx * 1e3,
            "m-", lw=2, label="source",
        )
    ax[2].plot(
        cx * cfg.dx * 1e3, cy * cfg.dx * 1e3,
        "c*", markersize=14, markeredgecolor="white", label="target",
    )
    ax[2].legend(loc="upper right", facecolor="black", labelcolor="white")
    ax[2].set_title(f"Peak pressure [Pa]  (max={p_max_arr.max():.2e})")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    for a in ax:
        a.set_xlabel("lateral [mm]")
        a.set_ylabel("axial [mm]")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_0000.png", dpi=120)
    print(f"[saved] {out_dir / 'sample_0000.npz'}  and  sample_0000.png")


if __name__ == "__main__":
    main()
