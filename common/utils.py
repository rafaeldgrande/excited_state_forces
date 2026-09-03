import numpy as np
import h5py

__all__ = [
    'ignore_0_freq_modes', 'FLAVOR_DESC',
    '_gb', '_downsample_idx', 'unpolarized_invariant', 'read_eqp_dat_file',
    'cartesian_from_bvec_basis', 'load_exciton_phonon_matrix',
    'isotropic_parallel', 'isotropic_perpendicular', 'depolarization_ratio',
]

ignore_0_freq_modes = True

# Flavor labels for resonant Raman calculations.
# Renumbered 2026-08-05 (IPA flavors moved to the front; "d2"/"d3" jargon
# renamed to explicit diagonal/off-diagonal language; new flavor 6, second-
# order double resonance alone, previously never offered on its own). See
# resonant_raman/README.md's "Raman Flavor Index" table for the full
# description of each combination and which files/directories they read from.
FLAVOR_DESC = {
    0: 'IPA first order',
    1: 'IPA second order',
    2: 'IPA first + second order',
    3: 'First order, diagonal exciton-phonon only',
    4: 'First order, diagonal + off-diagonal exciton-phonon',
    5: 'Second order, triple resonance only',
    6: 'Second order, double resonance only',
    7: 'Second order, double + triple resonance',
    8: 'First order (diag+offdiag) + second order (double+triple)',
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


def _raman_invariants(a):
    """
    Shared building blocks for the isotropic (powder-averaged) Raman
    invariants: mean polarizability alpha_bar, anisotropy^2 gamma2, and
    antisymmetric-anisotropy^2 delta2.

    a : (3, 3, ...) complex array — first two axes are Cartesian indices.
    Returns (alpha_bar, gamma2, delta2), each with the trailing shape of `a`.
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
    return alpha_bar, gamma2, delta2


def unpolarized_invariant(a):
    """
    Unpolarized Raman invariant  45|ᾱ|² + 7γ² + 5δ² = I_parallel^iso + I_perp^iso.

    a : (3, 3, ...) complex array — first two axes are Cartesian indices.
    Works for any trailing shape: (), (Nfreq,), (N, Nfreq), etc.
    Returns an array (or scalar) with the same trailing shape.
    """
    alpha_bar, gamma2, delta2 = _raman_invariants(a)
    return 45 * np.abs(alpha_bar)**2 + 7 * gamma2 + 5 * delta2


def isotropic_parallel(a):
    """
    Isotropic (powder-averaged) parallel-polarized Raman intensity
    I_parallel^iso = 45|ᾱ|² + 4γ², for a randomly-oriented sample (e.g.
    molecules) where theta is not meaningful. Same input convention as
    `unpolarized_invariant`.
    """
    alpha_bar, gamma2, _ = _raman_invariants(a)
    return 45 * np.abs(alpha_bar)**2 + 4 * gamma2


def isotropic_perpendicular(a):
    """
    Isotropic (powder-averaged) perpendicular-polarized Raman intensity
    I_perp^iso = 3γ² + 5δ². Same input convention as `unpolarized_invariant`.
    """
    _, gamma2, delta2 = _raman_invariants(a)
    return 3 * gamma2 + 5 * delta2


def depolarization_ratio(a):
    """
    Depolarization ratio rho = I_perp^iso / I_parallel^iso for a randomly
    oriented sample. rho ~ 0 for totally symmetric modes (e.g. benzene's
    a1g ring-breathing mode); rho -> 3/4 in the fully depolarized limit.
    Same input convention as `unpolarized_invariant`.
    """
    return isotropic_perpendicular(a) / isotropic_parallel(a)


def cartesian_from_bvec_basis(d_b1, d_b2, d_b3, bvec):
    """
    Convert dipole/position-operator components measured along the
    (generally non-orthogonal) reciprocal lattice unit vectors b1,b2,b3 --
    BerkeleyGW's default polarization basis when no `polarization` card is
    given in absorption.inp, per BSE/vmtxel.f90 + Common/mtxel_optical.f90 --
    into genuine Cartesian x,y,z components.

    Each input is d_bi = P . b_hat_i (P = true Cartesian vector, b_hat_i =
    unit vector along reciprocal lattice vector i). Stacking B_hat =
    [b_hat_1 | b_hat_2 | b_hat_3] (columns): d = B_hat^T . P, so
    P = (B_hat^T)^-1 . d.

    bvec : (3,3) array, columns = reciprocal lattice vectors in Cartesian
        directions (e.g. from eigenvectors.h5's mf_header/crystal/bvec --
        units of `blat` are fine, the scale cancels on normalization).
    d_b1, d_b2, d_b3 : arrays of matching shape (broadcastable), real or
        complex.

    Returns (d_x, d_y, d_z), same shape as the inputs. A no-op (returns the
    inputs unchanged, up to floating-point precision) when bvec's columns
    are already Cartesian-orthonormal.
    """
    b_hat = bvec / np.linalg.norm(bvec, axis=0, keepdims=True)
    transform = np.linalg.inv(b_hat.T)
    stacked = np.stack([d_b1, d_b2, d_b3], axis=0)
    cart = np.tensordot(transform, stacked, axes=(1, 0))
    return cart[0], cart[1], cart[2]


def load_exciton_phonon_matrix(h5_path, limit_Nexc=None):
    """
    Load an exciton-phonon coupling file (produced by
    assemble_exciton_phonon_coeffs.py or excited_forces.py) and return the
    dense Hermitian matrix g^{BA}_nu = <B|dH/dQ_nu|A> plus phonon
    frequencies, in the same convention used throughout resonant_raman/.

    Handles both known on-disk schemas:
      - "new format": exciton_pairs (Npairs, 2), forces/ph/RPA (Npairs,
        Nmodes) with F = -<A|dH|B> per pair; assembled into a dense
        (Nmodes, Nexc, Nexc) matrix (missing pairs assumed zero, Hermitian
        symmetry applied: exc_ph[:, j, i] = exc_ph[:, i, j].conj()).
      - "old format": rpa_offdiag, already a dense (Nmodes, Nexc, Nexc)
        array.

    Parameters
    ----------
    h5_path : str
        Path to the exciton-phonon coupling HDF5 file.
    limit_Nexc : int or None
        If given, truncate the exciton index to the first limit_Nexc states
        (for quick testing).

    Returns
    -------
    exc_ph : (Nmodes, Nexc, Nexc) complex ndarray
        exc_ph[nu, i, j] = <i|dH/dQ_nu|j>, eV/ang (or eV/ang^2 if built from
        a second-derivative run -- units are not converted here, only
        carried through from the source file).
    freqs_eV : (Nmodes,) ndarray or None
        Phonon frequencies in eV, if present in the file under
        system/phonon_frequencies (stored in cm^-1); None otherwise.
    """
    from .constants import rec_cm_to_eV

    freqs_eV = None
    with h5py.File(h5_path, 'r') as hf:
        if 'forces/ph/RPA' in hf:
            pairs = hf['exciton_pairs'][:]      # (Npairs, 2) 1-based
            forces = hf['forces/ph/RPA'][:]     # (Npairs, Nmodes)
            max_exc = int(pairs.max())
            _Nm = forces.shape[1]
            exc_ph = np.zeros((_Nm, max_exc, max_exc), dtype=complex)
            for k, (i, j) in enumerate(pairs):
                val = -forces[k]                # negate: F = -<A|dH|B> -> <A|dH|B>
                exc_ph[:, i - 1, j - 1] = val
                if i != j:
                    exc_ph[:, j - 1, i - 1] = val.conj()
        else:
            exc_ph = hf['rpa_offdiag'][:]
        if 'system/phonon_frequencies' in hf:
            freqs_eV = hf['system/phonon_frequencies'][:] * rec_cm_to_eV

    if limit_Nexc is not None:
        exc_ph = exc_ph[:, :limit_Nexc, :limit_Nexc]

    return exc_ph, freqs_eV


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
