import numpy as np

__all__ = [
    'ignore_0_freq_modes', 'FLAVOR_DESC',
    '_gb', '_downsample_idx', 'unpolarized_invariant', 'read_eqp_dat_file',
]

ignore_0_freq_modes = True

# Flavor labels for resonant Raman calculations
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

    data = np.loadtxt(eqp_file)

    Nbnds = int(data[0, 3])
    band_indexes = data[1:Nbnds+1, 1]

    Kpoints = data[0::Nbnds+1]
    Kpoints = Kpoints[:, :3]

    Nk = len(Kpoints)
    print(f'Number of kpoints {Nk}')

    for ibnd in range(Nbnds):
        temp = data[ibnd+1::Nbnds+1]
        bands_dft.append(temp[:, 2])
        bands_qp.append(temp[:, 3])

    return np.array(bands_dft), np.array(bands_qp), Kpoints, Nk, band_indexes
