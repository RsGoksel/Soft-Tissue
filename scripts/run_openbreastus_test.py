"""
End-to-end test with a REAL OpenBreastUS phantom, NATIVE sound-speed path.

Loads one 480x480 phantom (resampled to 256x256 for speed), inserts a tumor
directly into the continuous sound-speed map, derives density and
attenuation from the speed field, runs k-wave-python, and saves a 4-panel
preview showing:
    raw speed | speed + tumor | derived density | peak pressure

Run:
    python scripts/run_openbreastus_test.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phantom import insert_tumor_speed, load_openbreastus_speed   # noqa: E402
from simulate import SimConfig, run_focused_sim_from_speed        # noqa: E402


MAT_PATH = PROJECT_ROOT / "data" / "breast_test_speed.mat"
PHANTOM_IDX = 0      # which of the 800 phantoms to load
GRID_SIZE = 256      # resample from native 480 -> 256 for CPU speed


def main() -> None:
    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(0)

    # 1. load raw continuous sound-speed map (no quantisation)
    raw_speed = load_openbreastus_speed(
        MAT_PATH,
        phantom_index=PHANTOM_IDX,
        target_size=GRID_SIZE,
    )
    print(f"[phantom] loaded OpenBreastUS index={PHANTOM_IDX}, "
          f"speed range {raw_speed.min():.1f}..{raw_speed.max():.1f} m/s")

    # 2. insert tumor directly into the sound-speed field
    speed_with_tumor, (cy, cx, r) = insert_tumor_speed(raw_speed, rng=rng)
    print(f"[phantom] tumor at (axial={cy}, lateral={cx}), radius={r} px")

    # 3. native-path simulation
    cfg = SimConfig(use_gpu=False)
    result = run_focused_sim_from_speed(
        speed_with_tumor, focus_yx=(cy, cx), config=cfg,
    )
    print(f"[sim] p_max range: {result['p_max'].min():.2e} .. "
          f"{result['p_max'].max():.2e} Pa   "
          f"(focus gain ~{result['p_max'].max() / result['p_max'].mean():.0f}x)")

    # 4. save
    np.savez_compressed(
        out_dir / "obus_native_0000.npz",
        raw_speed=raw_speed,
        speed_with_tumor=speed_with_tumor,
        c=result["c"],
        rho=result["rho"],
        alpha=result["alpha"],
        p_max=result["p_max"],
        focus_yx=result["focus_yx"],
    )

    # 5. 4-panel preview (axes in mm)
    n_ax, n_lat = raw_speed.shape
    extent_mm = [0, n_lat * cfg.dx * 1e3, n_ax * cfg.dx * 1e3, 0]

    src_ys, src_xs = np.where(result["source_mask"])

    fig, ax = plt.subplots(1, 4, figsize=(17, 4.5))

    im0 = ax[0].imshow(raw_speed, cmap="viridis", extent=extent_mm)
    ax[0].set_title(f"Raw OpenBreastUS #{PHANTOM_IDX}\n(speed m/s)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)

    im1 = ax[1].imshow(speed_with_tumor, cmap="viridis", extent=extent_mm)
    ax[1].plot(
        cx * cfg.dx * 1e3, cy * cfg.dx * 1e3,
        "r+", markersize=14, markeredgewidth=2,
    )
    ax[1].set_title(f"Speed + tumor\n(tumor c=1550, r={r}px)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)

    im2 = ax[2].imshow(result["rho"], cmap="magma", extent=extent_mm)
    ax[2].set_title("Derived density [kg/m^3]")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)

    p_max_arr = result["p_max"]
    vmax = float(np.percentile(p_max_arr, 99.5))
    im3 = ax[3].imshow(p_max_arr, cmap="hot", vmin=0, vmax=vmax, extent=extent_mm)
    if src_ys.size:
        ax[3].plot(
            src_xs * cfg.dx * 1e3,
            src_ys * cfg.dx * 1e3,
            "m-", lw=2, label="source",
        )
    ax[3].plot(
        cx * cfg.dx * 1e3, cy * cfg.dx * 1e3,
        "c*", markersize=14, markeredgecolor="white", label="target",
    )
    ax[3].legend(loc="upper right", facecolor="black", labelcolor="white")
    ax[3].set_title(f"Peak pressure [Pa]\nmax={p_max_arr.max():.2e}")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)

    for a in ax:
        a.set_xlabel("lateral [mm]")
        a.set_ylabel("axial [mm]")

    fig.tight_layout()
    fig.savefig(out_dir / "obus_native_0000.png", dpi=120)
    print(f"[saved] {out_dir / 'obus_native_0000.npz'} "
          f"and obus_native_0000.png")


if __name__ == "__main__":
    main()
