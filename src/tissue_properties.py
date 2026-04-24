"""
Tissue label -> acoustic property mapping.

OA-Breast label convention:
    0 = background (assumed water/coupling medium)
    2 = fibroglandular tissue
    3 = fat
    4 = skin
    5 = vessel (blood)

We add:
    6 = tumor (inserted by us during augmentation)

Values are means from Gorsel1 Table 8 where available, and from standard
references (Mast 2000, Duck 1990, IT'IS database) for the rest.
Density [kg/m^3], sound speed [m/s], power-law attenuation alpha_0
[dB/(MHz^y cm)] with y ~ 1.5 for soft tissue.
"""

import numpy as np

BACKGROUND = 0
FIBROGLANDULAR = 2
FAT = 3
SKIN = 4
VESSEL = 5
TUMOR = 6

# (sound_speed [m/s], density [kg/m^3], alpha_0 [dB/(MHz^y cm)])
_PROPS = {
    BACKGROUND:     (1500.0, 1000.0, 0.002),   # water
    FIBROGLANDULAR: (1515.0, 1041.0, 0.75),
    FAT:            (1450.0,  950.0, 0.48),
    SKIN:           (1615.0, 1090.0, 0.35),
    VESSEL:         (1584.0, 1040.0, 0.20),
    TUMOR:          (1550.0, 1066.0, 0.79),    # Table 8 mean
}

ALPHA_POWER = 1.5  # power-law exponent y


def labels_to_maps(label_map: np.ndarray):
    """
    Convert an integer label map (Nx, Ny[, Nz]) into three float32 arrays of
    the same shape: (sound_speed, density, alpha_coeff).
    """
    c = np.empty(label_map.shape, dtype=np.float32)
    rho = np.empty(label_map.shape, dtype=np.float32)
    alpha = np.empty(label_map.shape, dtype=np.float32)

    # fill unknown labels with background as a safe default
    c.fill(_PROPS[BACKGROUND][0])
    rho.fill(_PROPS[BACKGROUND][1])
    alpha.fill(_PROPS[BACKGROUND][2])

    for label, (cv, rv, av) in _PROPS.items():
        mask = label_map == label
        if mask.any():
            c[mask] = cv
            rho[mask] = rv
            alpha[mask] = av

    return c, rho, alpha
