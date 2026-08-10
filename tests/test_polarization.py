"""
Tests for resonant_raman/polarization.py -- angle-resolved polarized Raman
frame construction and tensor contraction.

Unlike the other resonant_raman/*.py scripts, polarization.py has no
module-level argparse/script logic, so it can be imported directly (no
_partial_import trick needed, see test_susceptibility_tensors.py).
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# Import `common` (the real top-level package, via conftest.py's repo-root
# sys.path entry) BEFORE adding resonant_raman/ to the path -- that dir has
# its own deprecated `common.py` stub which would otherwise shadow it.
from common import (unpolarized_invariant, isotropic_parallel,
                    isotropic_perpendicular, depolarization_ratio)

RR_DIR = Path(__file__).parent.parent / 'resonant_raman'
sys.path.insert(0, str(RR_DIR))
from polarization import (build_frame, polarization_vectors,
                          contract_first_order, contract_second_order,
                          alpha_plane_first_order, contract_plane,
                          JONES_PLUS, JONES_MINUS)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _theta_grid(dtheta_deg=10.0):
    dtheta = np.deg2rad(dtheta_deg)
    Ntheta = int(round(2.0 * np.pi / dtheta))
    return np.arange(Ntheta) * dtheta


def _random_tensor_1st(Nmodes=5, Nfreq=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(3, 3, Nmodes, Nfreq)) + 1j * rng.normal(size=(3, 3, Nmodes, Nfreq))


def _random_tensor_2nd(Nmodes=4, Nfreq=5, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(3, 3, Nmodes, Nmodes, Nfreq))
            + 1j * rng.normal(size=(3, 3, Nmodes, Nmodes, Nfreq)))


def _default_frame_intensity_1st(alpha, e_s, e_i, weight):
    """I(theta) = |weight * contract(alpha, e_s, e_i)|^2, summed over modes."""
    M = contract_first_order(alpha, e_s, e_i)               # (Ntheta, Nmodes, Nfreq)
    return np.abs(weight[np.newaxis, :, np.newaxis] * M)**2  # (Ntheta, Nmodes, Nfreq)


# ─────────────────────────────────────────────────────────────
# 1. Cartesian regression -- the most important check
# ─────────────────────────────────────────────────────────────

def test_cartesian_regression_first_order():
    """At theta=0 with n_hat=z, e1=x (defaults): I_parallel == |alpha_xx|^2,
    I_perp == |alpha_yx|^2, to machine precision -- for the whole (Nmodes,Nfreq)
    array, mirroring what resonant_raman.py's raman_maps[0,0]/[1,0] would give."""
    alpha = _random_tensor_1st()
    e1, e2, n = build_frame([0, 0, 1])
    theta = _theta_grid()
    e_i, e_par, e_perp = polarization_vectors(theta, e1, e2)

    M_par  = contract_first_order(alpha, e_par,  e_i)
    M_perp = contract_first_order(alpha, e_perp, e_i)

    assert np.allclose(M_par[0],  alpha[0, 0])   # xx
    assert np.allclose(M_perp[0], alpha[1, 0])   # yx


def test_cartesian_regression_second_order():
    alpha2 = _random_tensor_2nd()
    e1, e2, n = build_frame([0, 0, 1])
    theta = _theta_grid()
    e_i, e_par, e_perp = polarization_vectors(theta, e1, e2)

    M_par  = contract_second_order(alpha2, e_par,  e_i)
    M_perp = contract_second_order(alpha2, e_perp, e_i)

    assert np.allclose(M_par[0],  alpha2[0, 0])
    assert np.allclose(M_perp[0], alpha2[1, 0])


# ─────────────────────────────────────────────────────────────
# 2. Periodicity: I(theta + pi) == I(theta)
# ─────────────────────────────────────────────────────────────

def test_periodicity_first_order():
    alpha = _random_tensor_1st()
    e1, e2, n = build_frame([0, 0, 1])
    theta = _theta_grid(dtheta_deg=5.0)   # Ntheta=72, evenly split by pi
    e_i, e_par, e_perp = polarization_vectors(theta, e1, e2)
    weight = np.ones(alpha.shape[2])

    for e_s in (e_par, e_perp):
        I = _default_frame_intensity_1st(alpha, e_s, e_i, weight)
        half = len(theta) // 2
        assert np.allclose(I[:half], I[half:2 * half], atol=1e-10)


def test_periodicity_second_order():
    alpha2 = _random_tensor_2nd()
    e1, e2, n = build_frame([0, 0, 1])
    theta = _theta_grid(dtheta_deg=5.0)
    e_i, e_par, e_perp = polarization_vectors(theta, e1, e2)

    for e_s in (e_par, e_perp):
        M = contract_second_order(alpha2, e_s, e_i)
        I = np.abs(M)**2
        half = len(theta) // 2
        assert np.allclose(I[:half], I[half:2 * half], atol=1e-10)


# ─────────────────────────────────────────────────────────────
# 3. Isotropy: in-plane tensor = c * identity_2x2 -> I_parallel constant,
#    I_perp == 0
# ─────────────────────────────────────────────────────────────

def test_isotropy_in_plane():
    c = 2.5 + 1.3j
    alpha = np.zeros((3, 3, 1, 1), dtype=complex)
    alpha[0, 0, 0, 0] = c
    alpha[1, 1, 0, 0] = c

    e1, e2, n = build_frame([0, 0, 1])
    theta = _theta_grid(dtheta_deg=15.0)
    e_i, e_par, e_perp = polarization_vectors(theta, e1, e2)

    M_par  = contract_first_order(alpha, e_par,  e_i)[:, 0, 0]
    M_perp = contract_first_order(alpha, e_perp, e_i)[:, 0, 0]

    assert np.allclose(np.abs(M_par)**2, np.abs(c)**2)
    assert np.allclose(M_perp, 0.0, atol=1e-12)


# ─────────────────────────────────────────────────────────────
# 4. Degenerate-mode gauge invariance: a random unitary mixing within a
#    degenerate phonon subspace leaves the SUMMED intensity unchanged.
# ─────────────────────────────────────────────────────────────

def test_degenerate_mode_gauge_invariance():
    rng = np.random.default_rng(7)
    Nfreq = 4
    # Two degenerate modes (indices 0,1) + one non-degenerate mode (index 2)
    alpha = _random_tensor_1st(Nmodes=3, Nfreq=Nfreq, seed=7)
    weight = np.ones(3)

    e1, e2, n = build_frame([0, 0, 1])
    theta = _theta_grid(dtheta_deg=20.0)
    e_i, e_par, e_perp = polarization_vectors(theta, e1, e2)

    def summed_degenerate_intensity(a):
        I = _default_frame_intensity_1st(a, e_par, e_i, weight)
        return I[:, 0, :] + I[:, 1, :]     # sum over the degenerate subspace

    I_before = summed_degenerate_intensity(alpha)

    # Random unitary (2x2) mixing of modes 0,1, applied to the mode axis
    U = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
    alpha_mixed = alpha.copy()
    sub = alpha[:, :, 0:2, :]                                  # (3,3,2,Nfreq)
    mixed = np.einsum('ij,abjf->abif', U, sub, optimize=True)
    alpha_mixed[:, :, 0:2, :] = mixed

    I_after = summed_degenerate_intensity(alpha_mixed)

    assert np.allclose(I_before, I_after, atol=1e-8)


# ─────────────────────────────────────────────────────────────
# 5. Frame independence: rotating n_hat and the tensor together by the same
#    rotation R leaves I_parallel(theta) unchanged.
# ─────────────────────────────────────────────────────────────

def test_frame_independence():
    rng = np.random.default_rng(11)
    alpha = _random_tensor_1st(Nmodes=2, Nfreq=3, seed=11)
    weight = np.ones(2)

    e1, e2, n = build_frame([0, 0, 1])
    theta = _theta_grid(dtheta_deg=20.0)
    e_i, e_par, _ = polarization_vectors(theta, e1, e2)
    I_ref = _default_frame_intensity_1st(alpha, e_par, e_i, weight)

    # Random 3x3 rotation matrix R (orthogonal, det=+1)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    R = Q

    n_rot = R @ np.array([0.0, 0.0, 1.0])
    alpha_rot = np.einsum('ai,bj,ijmf->abmf', R, R, alpha, optimize=True)

    e1r, e2r, nr = build_frame(n_rot, r_ref=R @ np.array([1.0, 0.0, 0.0]))
    e_i_r, e_par_r, _ = polarization_vectors(theta, e1r, e2r)
    I_rot = _default_frame_intensity_1st(alpha_rot, e_par_r, e_i_r, weight)

    assert np.allclose(I_ref, I_rot, atol=1e-8)


# ─────────────────────────────────────────────────────────────
# 6. Benzene physical check: a totally symmetric (isotropic-tensor) mode
#    gives rho ~ 0 under the isotropic-average invariants.
# ─────────────────────────────────────────────────────────────

def test_benzene_totally_symmetric_mode_depolarization_ratio():
    # a1g ring-breathing mode: alpha proportional to the identity
    # (fully isotropic polarizability derivative) -> gamma^2 = delta^2 = 0
    # -> rho = I_perp^iso / I_parallel^iso = 0.
    a = np.zeros((3, 3), dtype=complex)
    a[0, 0] = a[1, 1] = a[2, 2] = 3.7 + 0.2j
    rho = depolarization_ratio(a)
    assert rho == pytest.approx(0.0, abs=1e-12)

    # Consistency: isotropic_parallel + isotropic_perpendicular == unpolarized_invariant
    assert isotropic_parallel(a) + isotropic_perpendicular(a) == \
        pytest.approx(unpolarized_invariant(a))


def test_benzene_e2g_mode_depolarized_limit():
    # A purely anisotropic (traceless, off-diagonal-free) mode approaches
    # the depolarized limit rho -> 3/4.
    a = np.zeros((3, 3), dtype=complex)
    a[0, 0] = 1.0
    a[1, 1] = -1.0
    rho = depolarization_ratio(a)
    assert rho == pytest.approx(0.75, abs=1e-10)


# ─────────────────────────────────────────────────────────────
# Helicity-resolved Raman (HELICITY_RAMAN_SPEC.md sec. 7) -- these operate
# directly on bare (2,2) in-plane blocks via alpha_plane_*/contract_plane,
# the same primitives resonant_raman.py's --helicity path uses. The
# 'jones'/'propagation' convention swap lives in resonant_raman.py (not
# polarization.py, which is convention-agnostic); mirrored here as a local
# helper so the underlying primitives get exercised under both conventions.
# ─────────────────────────────────────────────────────────────

def _helicity_pairs(convention):
    """Reproduce resonant_raman.py's --helicity-convention swap logic."""
    if convention == 'jones':
        c_s_plus, c_s_minus = JONES_PLUS, JONES_MINUS
    else:
        c_s_plus, c_s_minus = JONES_MINUS, JONES_PLUS
    c_i_plus, c_i_minus = JONES_PLUS, JONES_MINUS
    return c_s_plus, c_s_minus, c_i_plus, c_i_minus


def _sigma_intensities(alpha_plane, convention):
    c_s_plus, c_s_minus, c_i_plus, c_i_minus = _helicity_pairs(convention)
    pp = np.abs(contract_plane(alpha_plane, c_s_plus,  c_i_plus))**2
    pm = np.abs(contract_plane(alpha_plane, c_s_plus,  c_i_minus))**2
    mp = np.abs(contract_plane(alpha_plane, c_s_minus, c_i_plus))**2
    mm = np.abs(contract_plane(alpha_plane, c_s_minus, c_i_minus))**2
    return pp, pm, mp, mm


# ─────────────────────────────────────────────────────────────
# 7.1 Selection rules: A1-like (proportional to identity) preserves
#     helicity (++/-- nonzero, +-/-+ exactly zero); E-like (both the
#     off-diagonal and the diag(c,-c) forms) flips helicity (opposite
#     pattern). Swapping to the 'propagation' convention swaps which
#     combination is zero, since it only relabels the scattered +/- axis.
# ─────────────────────────────────────────────────────────────

def test_helicity_selection_rule_a1_like():
    c = 1.7 - 0.4j
    alpha_plane = np.array([[c, 0], [0, c]], dtype=complex)

    pp, pm, mp, mm = _sigma_intensities(alpha_plane, 'jones')
    assert pp == pytest.approx(abs(c)**2)
    assert mm == pytest.approx(abs(c)**2)
    assert pm == pytest.approx(0.0, abs=1e-12)
    assert mp == pytest.approx(0.0, abs=1e-12)

    # 'propagation' only relabels the scattered +/- axis -> zero/nonzero swaps.
    pp2, pm2, mp2, mm2 = _sigma_intensities(alpha_plane, 'propagation')
    assert pp2 == pytest.approx(0.0, abs=1e-12)
    assert mm2 == pytest.approx(0.0, abs=1e-12)
    assert pm2 == pytest.approx(abs(c)**2)
    assert mp2 == pytest.approx(abs(c)**2)


@pytest.mark.parametrize('alpha_plane', [
    np.array([[0, 1.7 - 0.4j], [1.7 - 0.4j, 0]], dtype=complex),   # E_(1)-like
    np.array([[1.7 - 0.4j, 0], [0, -(1.7 - 0.4j)]], dtype=complex),  # E_(2)-like
])
def test_helicity_selection_rule_e_like(alpha_plane):
    c = 1.7 - 0.4j
    pp, pm, mp, mm = _sigma_intensities(alpha_plane, 'jones')
    assert pp == pytest.approx(0.0, abs=1e-12)
    assert mm == pytest.approx(0.0, abs=1e-12)
    assert pm == pytest.approx(abs(c)**2)
    assert mp == pytest.approx(abs(c)**2)

    pp2, pm2, mp2, mm2 = _sigma_intensities(alpha_plane, 'propagation')
    assert pp2 == pytest.approx(abs(c)**2)
    assert mm2 == pytest.approx(abs(c)**2)
    assert pm2 == pytest.approx(0.0, abs=1e-12)
    assert mp2 == pytest.approx(0.0, abs=1e-12)


# ─────────────────────────────────────────────────────────────
# 7.2 Linear-polarization closed forms for the same E-like blocks:
#     I_E(1)(theta) = 4(a^2+b^2) cos^2(theta) sin^2(theta)
#     I_E(2)(theta) = (a^2+b^2) cos^2(2*theta)
#     and their theta-independent sum a^2+b^2 (degenerate-E-mode isotropy,
#     closed-form version of test_degenerate_mode_gauge_invariance).
# ─────────────────────────────────────────────────────────────

def test_helicity_linear_closed_forms_and_sum_invariance():
    a, b = 0.9, -1.3
    c = a + 1j * b
    mag2 = a**2 + b**2
    alpha_e1 = np.array([[0, c], [c, 0]], dtype=complex)
    alpha_e2 = np.array([[c, 0], [0, -c]], dtype=complex)

    theta = _theta_grid(dtheta_deg=7.0)
    e_theta = np.stack([np.cos(theta), np.sin(theta)])   # (2, Ntheta)

    I_e1 = np.abs(contract_plane(alpha_e1, e_theta, e_theta))**2
    I_e2 = np.abs(contract_plane(alpha_e2, e_theta, e_theta))**2

    assert np.allclose(I_e1, 4 * mag2 * np.cos(theta)**2 * np.sin(theta)**2)
    assert np.allclose(I_e2, mag2 * np.cos(2 * theta)**2)
    assert np.allclose(I_e1 + I_e2, mag2)   # theta-independent


# ─────────────────────────────────────────────────────────────
# 7.3 Sum rule: I_++ + I_+- + I_-+ + I_-- == 2 * <I_parallel + I_perp>_theta.
#     Exact identity (Frobenius-norm invariance under the {e1,e2}->{e+,e-}
#     unitary change of basis on one hand, and the quadratic-form circular
#     average on the other) -- holds for ANY Ntheta >= 3 evenly-spaced
#     angles, not just asymptotically, so a coarse grid suffices. Run
#     against both a random 2x2 block and one with (Nmodes,Nfreq) trailing
#     axes, matching the shapes alpha_plane_first_order actually produces.
# ─────────────────────────────────────────────────────────────

def _sum_rule_check(alpha_plane, theta):
    e_theta = np.stack([np.cos(theta), np.sin(theta)])
    e_perp  = np.stack([-np.sin(theta), np.cos(theta)])

    I_par  = np.abs(contract_plane(alpha_plane, e_theta, e_theta))**2
    I_perp = np.abs(contract_plane(alpha_plane, e_perp,  e_theta))**2
    lhs = 2.0 * np.mean(I_par + I_perp, axis=0)

    pp, pm, mp, mm = _sigma_intensities(alpha_plane, 'jones')
    rhs = pp + pm + mp + mm
    return lhs, rhs


def test_helicity_sum_rule_bare_block():
    rng = np.random.default_rng(3)
    alpha_plane = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    theta = _theta_grid(dtheta_deg=30.0)   # Ntheta=12 -- exact for any N>=3
    lhs, rhs = _sum_rule_check(alpha_plane, theta)
    assert lhs == pytest.approx(rhs, rel=1e-10)


def test_helicity_sum_rule_with_mode_freq_axes():
    rng = np.random.default_rng(4)
    alpha = _random_tensor_1st(Nmodes=3, Nfreq=4, seed=4)
    e1, e2, n = build_frame([0, 0, 1])
    alpha_plane = alpha_plane_first_order(alpha, e1, e2)   # (2,2,Nmodes,Nfreq)

    theta = _theta_grid(dtheta_deg=20.0)   # Ntheta=18
    lhs, rhs = _sum_rule_check(alpha_plane, theta)
    assert np.allclose(lhs, rhs, rtol=1e-8)


# ─────────────────────────────────────────────────────────────
# 7.4 Reciprocity: I_+- == I_-+ holds exactly for a REAL symmetric in-plane
#     block (sigma_+- and sigma_-+ come out as exact complex conjugates of
#     each other -- equal magnitude), but is NOT guaranteed in general for a
#     complex (e.g. resonant/absorptive) block. Per the spec's explicit
#     caution, only the real-symmetric case is asserted; the general case is
#     computed as a logged diagnostic, not an equality assertion.
# ─────────────────────────────────────────────────────────────

def test_helicity_reciprocity_real_symmetric():
    rng = np.random.default_rng(5)
    m = rng.normal(size=(2, 2))
    alpha_plane = (m + m.T) / 2   # real, symmetric
    _, pm, mp, _ = _sigma_intensities(alpha_plane.astype(complex), 'jones')
    assert pm == pytest.approx(mp, rel=1e-10)


def test_helicity_reciprocity_diagnostic_general_complex(capsys):
    rng = np.random.default_rng(6)
    alpha_plane = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    diag = np.linalg.norm(alpha_plane - alpha_plane.T)
    _, pm, mp, _ = _sigma_intensities(alpha_plane, 'jones')
    print(f'reciprocity diagnostic: ||alpha - alpha^T|| = {diag:.4g}, '
          f'I_+- = {pm:.4g}, I_-+ = {mp:.4g}')
    # Not asserted equal (spec sec. 7.4): a generic complex block need not
    # be reciprocal. Just confirm the diagnostic itself is well-defined.
    assert np.isfinite(diag) and diag >= 0.0
    assert np.isfinite(pm) and np.isfinite(mp)
