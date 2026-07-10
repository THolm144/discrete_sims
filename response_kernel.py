#!/usr/bin/env python3
"""
response_kernel.py
===================
Single shared definition of the parametric (energy- and depth-dependent,
skewed) response kernel used both to FIT the empirical response matrix
(build_empirical_response_matrix.py) and to CONSTRUCT R at analysis time
(unfold_profile_analysis.py).

Living in one place is deliberate: previously the fit script and the
consumer script could silently disagree (fit script computed sigma_layer
and threw it away; consumer loaded a frozen .npy). Import this module from
both places so there is exactly one definition of "what the kernel is".
"""

import numpy as np
from scipy.stats import skewnorm

# Reference energy the sigma0 term is anchored to. sigma_E_slope describes
# how the kernel width grows/shrinks relative to this point. Keeping this a
# fixed constant (not fit) avoids a sigma0/E_REF degeneracy.
E_REF_GEV = 100.0

PARAM_NAMES = ["sigma0", "sigma_slope", "skew", "tail_frac", "tail_mult", "sigma_E_slope"]


def sigma_of_layer(j, n_true, energy_val, params):
    """Width of the response kernel centered at true-layer j, for a given
    incident energy (GeV). Floor-clipped so the kernel never collapses to
    zero width."""
    sigma0, sigma_slope, skew, tail_frac, tail_mult, sigma_E_slope = params
    e = max(float(energy_val), 1e-6) if energy_val is not None else E_REF_GEV
    sigma_j = (
        sigma0
        + sigma_slope * (j - n_true / 2.0)
        + sigma_E_slope * np.log(e / E_REF_GEV)
    )
    return max(sigma_j, 0.15)


def build_kernel_matrix(n_reco, pad_layers, params, energy_val=None):
    """
    params = [sigma0, sigma_slope, skew, tail_frac, tail_mult, sigma_E_slope]

    sigma(j, E) = sigma0 + sigma_slope*(j - n_true/2) + sigma_E_slope*log(E/E_REF)

    energy_val: incident energy in GeV for THIS matrix instance. Pass the
    actual per-energy-point value at call time -- do not cache a single R
    across energies.
    """
    sigma0, sigma_slope, skew, tail_frac, tail_mult, sigma_E_slope = params
    n_true = n_reco + 2 * pad_layers
    idx = np.arange(n_true)

    R_full = np.zeros((n_true, n_true))
    for j in idx:
        sigma_j = sigma_of_layer(j, n_true, energy_val, params)
        x = idx - j
        core = skewnorm.pdf(x, a=skew, loc=0.0, scale=sigma_j)
        tail = skewnorm.pdf(x, a=skew, loc=0.0, scale=sigma_j * max(tail_mult, 1.0))
        col = (1.0 - tail_frac) * core + tail_frac * tail
        s = col.sum()
        if s > 0:
            col /= s
        R_full[:, j] = col

    return R_full[pad_layers: pad_layers + n_reco, :]