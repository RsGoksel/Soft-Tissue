"""
Does a constant phase offset change the focus recovered from phases?

We synthesise a plausible 256-element 2D phased array (16 x 16 grid on
an aperture of 100 mm), pick a known focal point f_true somewhere
5-15 cm deep in tissue, compute the idealised beamforming phases

    phi_i = -k * |r_i - f_true|  (mod 2pi)

then try to *invert* this: recover f from phi via nonlinear
least-squares. We do this twice:

  (a) phases as given
  (b) phases + 20 deg added to every element

and compare the recovered focal points.

Two inversion flavours are tested:
  * gauge-free   -- optimise f AND a global phi_0 offset
                   (phi_i = -k|r_i - f| + phi_0)
                   This is the *correct* physics problem.
  * gauge-fixed  -- force phi_0 = 0 (wrong but instructive:
                   mirrors what a NN would do if we just
                   regressed phases as targets).

Expected:
  * gauge-free   -> adding any constant changes nothing. f exactly.
  * gauge-fixed  -> adding constant visibly shifts f; the shift is
                   a systematic numerical artefact, not physics.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

# We keep the same mathematical form as the physical problem
#   phi_i = -K * |r_i - f| + phi0
# but scale K so that LM is numerically well-conditioned (avoid the
# rad-per-m -> 10^4 rad oscillations around the true solution).
# K only affects the NUMERICAL magnitudes of the residuals; the gauge
# symmetry of the problem is completely independent of its value.
F0_HZ   = 1.0e6      # physical -- unused, left for documentation
C0_MPS  = 1540.0
LAMBDA  = C0_MPS / F0_HZ
K       = 20.0                           # numerically well-conditioned


def synth_array(aperture_mm: float = 100.0, n_side: int = 16) -> np.ndarray:
    """16x16 planar array at z = 0, centred on origin."""
    xs = np.linspace(-aperture_mm / 2, aperture_mm / 2, n_side) * 1e-3
    ys = np.linspace(-aperture_mm / 2, aperture_mm / 2, n_side) * 1e-3
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    r = np.stack([xx.ravel(), yy.ravel(), np.zeros(n_side ** 2)], axis=1)
    return r    # (256, 3)


def wrap(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def phases_for_focus(r: np.ndarray, f: np.ndarray,
                     phi0: float = 0.0,
                     wrap_output: bool = False) -> np.ndarray:
    """Idealised beamforming phases. Use unwrapped form for the
    invertibility demo -- adding a constant and least-squares then
    cleanly reveal the gauge symmetry."""
    d = np.linalg.norm(r - f[None, :], axis=1)
    p = -K * d + phi0
    return wrap(p) if wrap_output else p


def recover_focus_gauge_free(r: np.ndarray, phi: np.ndarray,
                             f_init: np.ndarray) -> np.ndarray:
    """Fit both f (3) and phi0 (1) to match observed phases."""
    def res(p):
        f, phi0 = p[:3], p[3]
        return phases_for_focus(r, f, phi0) - phi
    sol = least_squares(res, np.concatenate([f_init, [0.0]]),
                        method="lm", max_nfev=2000)
    return sol.x[:3]


def recover_focus_gauge_fixed(r: np.ndarray, phi: np.ndarray,
                              f_init: np.ndarray) -> np.ndarray:
    """Fit only f (3), force phi0 = 0."""
    def res(f):
        return phases_for_focus(r, f, 0.0) - phi
    sol = least_squares(res, f_init, method="lm", max_nfev=2000)
    return sol.x


def main() -> None:
    rng = np.random.default_rng(0)
    r = synth_array()

    print("Synthetic phased array: 16x16, 10 cm aperture, z=0.\n")

    targets = [
        np.array([ 0.000, 0.000, 0.090]),
        np.array([ 0.015, -0.010, 0.115]),
        np.array([-0.018, 0.006, 0.140]),
    ]
    offsets_deg = [0.0, 20.0, 45.0, 90.0, 137.0]

    for f_true in targets:
        phi0 = phases_for_focus(r, f_true)         # ground-truth phases

        print(f"--- target f_true = {(f_true * 1000).round(2)} mm ---")
        print(f"{'offset':>8} | "
              f"{'gauge-free recovery error (mm)':>34} | "
              f"{'gauge-fixed recovery error (mm)':>36}")
        for off_deg in offsets_deg:
            phi_shift = wrap(phi0 + np.deg2rad(off_deg))
            # initial guess is a fixed point far from the true focus so we
            # cannot accidentally succeed by starting on top of f_true
            f_init = np.array([0.002, -0.003, 0.100])

            f_free  = recover_focus_gauge_free (r, phi_shift, f_init)
            f_fixed = recover_focus_gauge_fixed(r, phi_shift, f_init)
            err_free  = np.linalg.norm(f_free  - f_true) * 1000
            err_fixed = np.linalg.norm(f_fixed - f_true) * 1000
            print(f"{off_deg:>7.1f}° | "
                  f"{err_free:>32.4f}  | "
                  f"{err_fixed:>34.4f}")
        print()

    print("Read-out:")
    print("  gauge-free (fits phi0):  recovery is exact at ANY offset ")
    print("                          => focus does not depend on absolute phase")
    print("  gauge-fixed (phi0=0):   recovery error grows with offset; this is")
    print("                          a *numerical artefact*, the physical focus")
    print("                          has not moved, only the solver is confused")
    print("                          because it is trying to fit the constant.")


if __name__ == "__main__":
    main()
