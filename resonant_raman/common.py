
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
rec_cm_to_eV = 1.239841984e-4    # cm^-1 to eV
k_B          = 8.617333262145e-5  # Boltzmann constant in eV/K
hbar         = 6.582119569e-16   # reduced Planck constant in eV*s
ry2eV        = 13.605693122994   # Rydberg to eV
eV2ry        = 1.0 / ry2eV
TOL_ZERO     = 1e-6              # degeneracy threshold

ignore_0_freq_modes = True

# ---------------------------------------------------------------------------
# Flavor labels (must be consistent across all scripts)
# ---------------------------------------------------------------------------
FLAVOR_DESC = {
    0: 'First-order d2 only',
    1: 'First-order d3 only',
    2: 'Second-order triple resonance only',
    3: 'Second-order triple + double resonance',
    4: 'Second-order triple resonance + first-order d3',
    5: 'Second-order triple + double resonance + first-order d3',
    6: 'IPA first order',
    7: 'IPA second order',
    8: 'IPA first + second order',
}

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _gb(*shapes_and_dtypes):
    """Sum of array sizes in GB. Args: alternating (shape_tuple, dtype) pairs."""
    total = 0
    for shape, dtype in zip(shapes_and_dtypes[::2], shapes_and_dtypes[1::2]):
        total += np.prod(shape) * np.dtype(dtype).itemsize
    return total / 1024**3


def _downsample_idx(n_full, n_target):
    """Return indices for uniform down-sampling from n_full to n_target points."""
    return np.round(np.linspace(0, n_full - 1, n_target)).astype(int)


def unpolarized_invariant(a):
    """
    Unpolarized Raman invariant  45|ᾱ|² + 7γ² + 5δ².

    a : (3, 3, ...) complex array — first two axes are Cartesian indices.
    Works for any trailing shape: (), (Nfreq,), (N, Nfreq), etc.
    Returns an array (or scalar) with the same trailing shape.
    """
    alpha_bar = (a[0, 0] + a[1, 1] + a[2, 2]) / 3.0
    gamma2 = (0.5 * (np.abs(a[0, 0] - a[1, 1])**2 +
                     np.abs(a[1, 1] - a[2, 2])**2 +
                     np.abs(a[2, 2] - a[0, 0])**2) +
              3/4 * (np.abs(a[0, 1] + a[1, 0])**2 +
                     np.abs(a[0, 2] + a[2, 0])**2 +
                     np.abs(a[1, 2] + a[2, 1])**2))
    delta2 = 3/4 * (np.abs(a[0, 1] - a[1, 0])**2 +
                    np.abs(a[0, 2] - a[2, 0])**2 +
                    np.abs(a[1, 2] - a[2, 1])**2)
    return 45 * np.abs(alpha_bar)**2 + 7 * gamma2 + 5 * delta2

def read_eqp_dat_file(eqp_file):
    
    bands_dft, bands_qp = [], []
    
    # loading file
    data = np.loadtxt(eqp_file)

    # first getting number of bands in this file. first line is:   0.000000000  0.000000000  0.000000000      13
    Nbnds = int(data[0, 3])
    
    # getting list of band indexes
    band_indexes = data[1:Nbnds+1, 1]
    
    # now we get the k points in this file
    Kpoints = data[0::Nbnds+1] # get lines 0, Nbnds+1, 2*(Nbnds+2), ...
    Kpoints = Kpoints[:, :3] # remove last collumn with 
    
    Nk = len(Kpoints)
    print(f'Number of kpoints {Nk}')

    for ibnd in range(Nbnds):
        temp = data[ibnd+1::Nbnds+1]
        bands_dft.append(temp[:, 2])
        bands_qp.append(temp[:, 3])
                
    return np.array(bands_dft), np.array(bands_qp), Kpoints, Nk, band_indexes