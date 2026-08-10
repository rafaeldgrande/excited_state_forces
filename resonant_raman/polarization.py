"""
Frame construction and Cartesian-tensor contraction helpers for angle-resolved
polarized Raman intensities I_parallel(theta), I_perp(theta).

See resonant_raman/README.md's "Polarized Raman" subsection for the full
theory. Key point: contract Cartesian indices with the incident/scattered
polarization vectors BEFORE squaring (`resonant_raman.py`'s existing
Cartesian path squares each (alpha,beta) component individually, which
discards the relative phases responsible for angular structure -- these
functions preserve those phases).

Pure functions, no module-level script logic -- safe to `import` directly
(unlike the other resonant_raman/*.py scripts, which run argparse at import
time).
"""
import numpy as np


def build_frame(n_hat, r_ref=None):
    """Right-handed orthonormal triad (e1, e2, n_hat) spanning the scattering plane.

    n_hat : (3,) scattering-plane normal (backscattering propagation direction).
        Normalized internally; must be non-zero.
    r_ref : (3,) optional reference vector fixing the theta=0 direction.
        Defaults to x-hat, falling back to y-hat if that's within ~25 deg of
        n_hat (|r . n_hat| > 0.9).

    Returns (e1, e2, n), each (3,) unit vectors.
    """
    n = np.asarray(n_hat, dtype=float)
    if np.linalg.norm(n) < 1e-12:
        raise ValueError('--n-hat must be a non-zero vector')
    n = n / np.linalg.norm(n)

    if r_ref is None:
        r = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(r, n)) > 0.9:
            r = np.array([0.0, 1.0, 0.0])
    else:
        r = np.asarray(r_ref, dtype=float)

    e1 = r - np.dot(r, n) * n
    if np.linalg.norm(e1) < 1e-8:
        raise ValueError('--theta-ref is (nearly) parallel to --n-hat')
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return e1, e2, n


def polarization_vectors(theta, e1, e2):
    """Incident and scattered (parallel/perpendicular) polarization vectors.

    theta : (Ntheta,) angles in radians.
    e1, e2 : (3,) orthonormal in-plane basis from build_frame().

    Returns (e_i, e_par, e_perp), each (3, Ntheta) real arrays.
    e_par is identical to e_i (parallel-analyzer geometry); kept as a
    separate name for readability at call sites.
    """
    c, s = np.cos(theta), np.sin(theta)
    e_i    = np.outer(e1, c) + np.outer(e2, s)
    e_perp = -np.outer(e1, s) + np.outer(e2, c)
    return e_i, e_i, e_perp


def contract_first_order(alpha, e_s, e_i):
    """Contract a first-order susceptibility tensor with polarization vectors.

    alpha : (3, 3, Nmodes, Nfreq) complex.
    e_s, e_i : (3, Ntheta) real (or complex, for future circular polarizations).

    Returns M : (Ntheta, Nmodes, Nfreq) complex.
    """
    return np.einsum('at,abmf,bt->tmf', e_s.conj(), alpha, e_i, optimize=True)


def contract_second_order(alpha2, e_s, e_i):
    """Contract a second-order susceptibility tensor with polarization vectors.

    alpha2 : (3, 3, Nmodes, Nmodes, Nfreq) complex.
    e_s, e_i : (3, Ntheta) real (or complex, for future circular polarizations).

    Returns M : (Ntheta, Nmodes, Nmodes, Nfreq) complex.
    """
    return np.einsum('at,abmnf,bt->tmnf', e_s.conj(), alpha2, e_i, optimize=True)


# ── Helicity-resolved Raman (HELICITY_RAMAN_SPEC.md) ───────────────────────────
#
# The contraction e_s^dagger . alpha . e_i is bilinear in the polarization
# vectors, and every polarization of interest (linear at any theta, circular,
# elliptical) lies in the {e1,e2} plane -- so the entire polarization
# dependence is carried by the small 2x2 in-plane block
#   alpha_plane[j,k] = e_j^dagger . alpha . e_k,   j,k in {0,1} <-> {e1,e2}.
# This is ~25x smaller than the theta-resolved M_parallel/M_perp arrays at
# the default theta grid, and makes every fixed-polarization observable
# (circular included) a cheap 2-vector contraction instead of a full (3,3,...)
# tensor contraction. Note: `contract_first_order`/`contract_second_order`
# above are NOT changed by this addition -- they still operate correctly on
# the full 3D tensor (e_i/e_par/e_perp already lie in the {e1,e2} plane
# despite being expressed in 3D coordinates), so the already-validated
# theta-swept linear path (I_parallel/I_perp) is untouched. alpha_plane is
# used for the new, additional fixed-polarization observables below.

# Jones vectors for circular polarization, in the {e1,e2} in-plane basis
# (2-vectors -- combine with build_frame()'s e1,e2 via alpha_plane_*, not the
# full 3D e_s/e_i used by contract_first_order/contract_second_order).
JONES_PLUS  = np.array([1.0, 1.0j]) / np.sqrt(2)
JONES_MINUS = np.array([1.0, -1.0j]) / np.sqrt(2)


def alpha_plane_first_order(alpha, e1, e2):
    """Project a first-order susceptibility tensor onto the {e1,e2} in-plane block.

    alpha : (3, 3, Nmodes, Nfreq) complex.
    e1, e2 : (3,) orthonormal in-plane basis from build_frame().

    Returns alpha_plane : (2, 2, Nmodes, Nfreq) complex.
    """
    E = np.stack([e1, e2])   # (2, 3)
    return np.einsum('ja,abmf,kb->jkmf', E.conj(), alpha, E, optimize=True)


def alpha_plane_second_order(alpha2, e1, e2):
    """Project a second-order susceptibility tensor onto the {e1,e2} in-plane block.

    alpha2 : (3, 3, Nmodes, Nmodes, Nfreq) complex.
    e1, e2 : (3,) orthonormal in-plane basis from build_frame().

    Returns alpha_plane : (2, 2, Nmodes, Nmodes, Nfreq) complex.
    """
    E = np.stack([e1, e2])   # (2, 3)
    return np.einsum('ja,abmnf,kb->jkmnf', E.conj(), alpha2, E, optimize=True)


def contract_plane(alpha_plane, c_s, c_i):
    """Contract an in-plane block with a pair of 2-vector polarizations.

    alpha_plane : (2, 2, Nmodes, [Nmodes,] Nfreq) complex, from
        alpha_plane_first_order/alpha_plane_second_order.
    c_s, c_i : complex 2-vectors, EITHER shape (2,) for a single fixed
        polarization pair (e.g. JONES_PLUS/JONES_MINUS for helicity), OR
        shape (2, Nconf) to evaluate several configurations at once (e.g.
        (cos(theta), sin(theta)) stacks for the linear theta-sweep, though
        contract_first_order/contract_second_order already cover that case
        directly on the full 3D tensor).

    Returns M with shape (Nmodes, [Nmodes,] Nfreq) for a single pair, or
    (Nconf, Nmodes, [Nmodes,] Nfreq) for a stack.
    """
    c_s = np.asarray(c_s)
    c_i = np.asarray(c_i)
    if c_s.ndim == 1:
        return np.einsum('j,jk...,k->...', c_s.conj(), alpha_plane, c_i, optimize=True)
    return np.einsum('jt,jk...,kt->t...', c_s.conj(), alpha_plane, c_i, optimize=True)
