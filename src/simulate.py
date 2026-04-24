"""
Minimal 2D k-wave-python simulation of a focused ultrasound source into a
heterogeneous tissue map. Records the peak-pressure field over the grid
(the quantity we want the FNO/DeepONet to learn).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.filters import smooth
from kwave.utils.mapgen import make_disc
from kwave.utils.signals import tone_burst

from tissue_properties import ALPHA_POWER, labels_to_maps


# --------------------------------------------------------------------------- #
# Continuous (c, rho, alpha) model used by the NATIVE sound-speed path.
# Fitted against the reference breast tissue values in tissue_properties.py:
#     fat        : c=1450  rho= 950
#     water/bg   : c=1500  rho=1000
#     fibro      : c=1515  rho=1041
#     tumour ref : c=1550  rho=1066
#     skin       : c=1615  rho=1090
# A simple linear regression rho = 0.85 * c - 282 reproduces these within
# <5% and matches Mast (2000) for soft tissue -- good enough for operator
# learning, avoids the staircase artefacts that quantisation causes.
# --------------------------------------------------------------------------- #
def speed_to_density(c_map: np.ndarray) -> np.ndarray:
    return (0.85 * c_map - 282.0).astype(np.float32)


def speed_to_alpha(c_map: np.ndarray) -> np.ndarray:
    """
    Smooth piece-wise attenuation model. Fatty regions (low c) attenuate
    the least; fibroglandular peaks around 1510 m/s; skin (very high c) is
    less attenuating than fibro. Implemented as a smooth mixture of two
    Gaussians around the fibroglandular peak so there are no step jumps.
    """
    base = 0.45
    peak = 0.35 * np.exp(-((c_map - 1515.0) ** 2) / (2.0 * 30.0 ** 2))
    return (base + peak).astype(np.float32)


@dataclass
class SimConfig:
    dx: float = 0.2e-3          # grid spacing [m]  (0.2 mm -> ~7.5 PPW at 1 MHz in water)
    source_f0: float = 1.0e6    # source frequency [Hz]
    source_cycles: int = 5
    source_amp: float = 1.0e6   # source pressure amplitude [Pa]
    pml_size: int = 20
    cfl: float = 0.3
    t_end: float = 80e-6        # long enough for wave to cross the padded grid
    use_gpu: bool = True
    source_axial_offset: int = 15   # grid points from the top of the PADDED domain
    aperture_frac: float = 0.80     # aperture width as a fraction of the lateral extent
    water_standoff_mm: float = 12.0 # how much water padding to add ABOVE the phantom
    water_speed: float = 1500.0     # m/s -- used to fill the standoff padding


def _pad_water_standoff(
    c_map: np.ndarray,
    rho_map: np.ndarray,
    alpha_map: np.ndarray,
    focus_yx: tuple[int, int],
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int], int]:
    """
    Prepend rows of water to the top of each medium field so the source has a
    clear standoff before entering tissue. Returns the padded maps, the
    shifted focus index, and the pad size in grid points.
    """
    pad = int(round(cfg.water_standoff_mm * 1e-3 / cfg.dx))
    if pad <= 0:
        return c_map, rho_map, alpha_map, focus_yx, 0

    n_lat = c_map.shape[1]
    water_c = np.full((pad, n_lat), cfg.water_speed, dtype=c_map.dtype)
    water_rho = np.full((pad, n_lat), 1000.0, dtype=rho_map.dtype)
    water_alpha = np.full((pad, n_lat), 0.002, dtype=alpha_map.dtype)

    c_padded = np.concatenate([water_c, c_map], axis=0)
    rho_padded = np.concatenate([water_rho, rho_map], axis=0)
    alpha_padded = np.concatenate([water_alpha, alpha_map], axis=0)

    shifted_focus = (int(focus_yx[0]) + pad, int(focus_yx[1]))
    return c_padded, rho_padded, alpha_padded, shifted_focus, pad


def _run_with_medium(
    c_map: np.ndarray,
    rho_map: np.ndarray,
    alpha_map: np.ndarray,
    focus_yx: tuple[int, int],
    cfg: SimConfig,
) -> dict:
    """Shared k-Wave driver. All input paths converge here."""
    # 1) add water standoff ABOVE the phantom so the source has clear water
    c_map, rho_map, alpha_map, focus_yx, pad = _pad_water_standoff(
        c_map, rho_map, alpha_map, focus_yx, cfg,
    )

    n_axial, n_lateral = c_map.shape
    kgrid = kWaveGrid([n_axial, n_lateral], [cfg.dx, cfg.dx])

    medium = kWaveMedium(
        sound_speed=c_map,
        density=rho_map,
        alpha_coeff=alpha_map,
        alpha_power=ALPHA_POWER,
    )
    kgrid.makeTime(c_map.max(), cfl=cfg.cfl, t_end=cfg.t_end)

    focus_axial, focus_lateral = int(focus_yx[0]), int(focus_yx[1])
    axial_src = cfg.source_axial_offset

    aperture_half = int(cfg.aperture_frac * n_lateral / 2)
    lat_lo = max(0, focus_lateral - aperture_half)
    lat_hi = min(n_lateral, focus_lateral + aperture_half)

    source = kSource()
    source_mask = np.zeros((n_axial, n_lateral), dtype=bool)
    source_mask[axial_src, lat_lo:lat_hi] = True
    source.p_mask = source_mask

    c0 = float(np.mean(c_map))
    lateral_positions = np.arange(lat_lo, lat_hi)
    dists = np.sqrt(
        ((lateral_positions - focus_lateral) * cfg.dx) ** 2
        + ((axial_src - focus_axial) * cfg.dx) ** 2
    )
    delays = (dists.max() - dists) / c0
    delay_samples = np.round(delays / kgrid.dt).astype(int)

    base = tone_burst(1.0 / kgrid.dt, cfg.source_f0, cfg.source_cycles).squeeze()
    n_src = lateral_positions.size
    sig_len = int(base.size + delay_samples.max() + 10)
    source_p = np.zeros((n_src, sig_len), dtype=np.float32)
    for i, d in enumerate(delay_samples):
        source_p[i, d : d + base.size] = cfg.source_amp * base
    source.p = source_p

    sensor = kSensor()
    sensor.mask = np.ones((n_axial, n_lateral), dtype=bool)
    sensor.record = ["p_max_all"]

    sim_opts = SimulationOptions(
        pml_size=cfg.pml_size,
        pml_inside=False,
        data_cast="single",
        save_to_disk=True,
    )
    exec_opts = SimulationExecutionOptions(is_gpu_simulation=cfg.use_gpu)

    out = kspaceFirstOrder2D(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=sim_opts,
        execution_options=exec_opts,
    )
    p_max = np.asarray(out["p_max_all"]).reshape(n_axial, n_lateral).astype(np.float32)

    return {
        "c": c_map,
        "rho": rho_map,
        "alpha": alpha_map,
        "p_max": p_max,
        "focus_yx": np.asarray((focus_axial, focus_lateral), dtype=np.int32),
        "source_mask": source_mask,
        "pad": int(pad),
    }


def run_focused_sim_from_speed(
    speed_map: np.ndarray,
    focus_yx: tuple[int, int],
    config: Optional[SimConfig] = None,
) -> dict:
    """
    NATIVE sound-speed path. Skips label quantisation entirely -- the caller
    supplies a continuous c(x,y) field (e.g. an OpenBreastUS phantom) and
    density + attenuation are derived from it via `speed_to_density` and
    `speed_to_alpha`. This preserves the full heterogeneity of the source
    data and avoids staircase refraction artefacts.
    """
    cfg = config or SimConfig()
    c_map = speed_map.astype(np.float32, copy=True)
    rho_map = speed_to_density(c_map)
    alpha_map = speed_to_alpha(c_map)
    return _run_with_medium(c_map, rho_map, alpha_map, focus_yx, cfg)


def run_focused_sim(
    label_map: np.ndarray,
    focus_yx: tuple[int, int],
    config: Optional[SimConfig] = None,
) -> dict:
    """
    LABEL path. `label_map` has shape (n_axial, n_lateral). Discrete tissue
    labels are mapped to (c, rho, alpha) via tissue_properties.labels_to_maps.
    Use this for the synthetic phantom; prefer `run_focused_sim_from_speed`
    for OpenBreastUS data to keep the continuous heterogeneity.
    """
    cfg = config or SimConfig()
    c_map, rho_map, alpha_map = labels_to_maps(label_map)
    out = _run_with_medium(c_map, rho_map, alpha_map, focus_yx, cfg)
    out["labels"] = label_map.astype(np.uint8)
    return out
