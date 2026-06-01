"""
Tests for excited_forces_m.py:
  - correct_comp_vector
  - find_kpoint
  - one_over_E_diff
  - apply_Qshift_on_valence_states
  - read_exciton_pairs
  - Vectorized vs non-vectorized consistency for force matrix elements
"""
import os
import pytest
import numpy as np
from excited_forces_m import (
    correct_comp_vector,
    find_kpoint,
    one_over_E_diff,
    apply_Qshift_on_valence_states,
    read_exciton_pairs,
    compute_A_dRPA_dr_imode_B_vectorized,
    compute_A_dRPA_dr_imode_B_not_vectorized,
    compute_A_dRPAdiag_dr_imode_B_vectorized,
    compute_A_dRPAdiag_dr_imode_B_not_vectorized,
    compute_A_dKernel_dr_imode_B_vectorized,
    compute_A_dKernel_dr_imode_B_not_vectorized,
    calc_Dkernel_vectorized,
    calc_Dkernel_single_mode_not_vectorized,
)


# ─────────────────────────────────────────────────────────────
# correct_comp_vector
# ─────────────────────────────────────────────────────────────

class TestCorrectCompVector:
    def test_zero(self):
        assert correct_comp_vector(0.0) == pytest.approx(0.0)

    def test_half(self):
        assert correct_comp_vector(0.5) == pytest.approx(0.5)

    def test_one_folds_to_zero(self):
        assert correct_comp_vector(1.0) == pytest.approx(0.0)

    def test_negative_half_maps_to_half(self):
        assert correct_comp_vector(-0.5) == pytest.approx(0.5)

    def test_one_and_half_folds_to_half(self):
        assert correct_comp_vector(1.5) == pytest.approx(0.5)

    def test_negative_small(self):
        assert correct_comp_vector(-0.1) == pytest.approx(0.9)

    def test_large_positive(self):
        assert correct_comp_vector(3.7) == pytest.approx(0.7)

    def test_two_folds_to_zero(self):
        assert correct_comp_vector(2.0) == pytest.approx(0.0)

    def test_result_always_in_0_1(self):
        for x in np.linspace(-3.0, 3.0, 61):
            result = correct_comp_vector(x)
            assert 0.0 <= result < 1.0 + 1e-9, f"Out of range for input {x}: {result}"


# ─────────────────────────────────────────────────────────────
# find_kpoint
# ─────────────────────────────────────────────────────────────

class TestFindKpoint:
    def test_exact_match_first_entry(self):
        K_list = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        assert find_kpoint(np.array([0.0, 0.0, 0.0]), K_list) == 0

    def test_exact_match_second_entry(self):
        K_list = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        assert find_kpoint(np.array([0.5, 0.0, 0.0]), K_list) == 1

    def test_not_found_returns_minus_one(self):
        K_list = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        assert find_kpoint(np.array([0.3, 0.0, 0.0]), K_list) == -1

    def test_bz_translation_by_one(self):
        # A k-point shifted by a full reciprocal lattice vector should match
        K_list = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
        # [1, 0, 0] - round([1, 0, 0]) = [0, 0, 0] → norm 0
        assert find_kpoint(np.array([1.0, 0.0, 0.0]), K_list) == 0

    def test_bz_translation_negative(self):
        K_list = np.array([[0.5, 0.0, 0.0]])
        # [-0.5, 0, 0] - round([-0.5, 0, 0]) = 0 or 0.5 depending on rounding,
        # but diff = [-0.5, 0, 0] - [0.5, 0, 0] = [-1, 0, 0]
        # then -1 - round(-1) = 0 → norm 0
        assert find_kpoint(np.array([-0.5, 0.0, 0.0]), K_list) == 0

    def test_returns_last_occurrence(self):
        # find_kpoint does not break early, so it returns the last matching index
        K_list = np.array([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0], [0.1, 0.0, 0.0]])
        assert find_kpoint(np.array([0.1, 0.0, 0.0]), K_list) == 2

    def test_empty_list(self):
        K_list = np.zeros((0, 3))
        assert find_kpoint(np.array([0.0, 0.0, 0.0]), K_list) == -1


# ─────────────────────────────────────────────────────────────
# one_over_E_diff
# ─────────────────────────────────────────────────────────────

class TestOneOverEDiff:
    def test_diagonal_is_zero(self):
        Ebands = np.array([[0.0, 1.0, 2.0]])
        result = one_over_E_diff(Ebands)
        for ib in range(3):
            assert result[0, ib, ib] == pytest.approx(0.0)

    def test_off_diagonal_inverse(self):
        Ebands = np.array([[0.0, 2.0]])
        result = one_over_E_diff(Ebands)
        assert result[0, 0, 1] == pytest.approx(-0.5)   # 1/(0-2) = -0.5
        assert result[0, 1, 0] == pytest.approx(0.5)    # 1/(2-0) = 0.5

    def test_degenerate_bands_give_zero(self):
        Ebands = np.array([[1.0, 1.0]])  # degenerate
        result = one_over_E_diff(Ebands)
        assert result[0, 0, 1] == pytest.approx(0.0)
        assert result[0, 1, 0] == pytest.approx(0.0)

    def test_antisymmetry(self):
        Ebands = np.arange(6, dtype=float).reshape(2, 3) + 0.1
        result = one_over_E_diff(Ebands)
        # result[ik, ib1, ib2] = -result[ik, ib2, ib1]
        assert np.allclose(result, -result.transpose(0, 2, 1))

    def test_output_shape(self):
        Ebands = np.ones((4, 5))
        result = one_over_E_diff(Ebands)
        assert result.shape == (4, 5, 5)

    def test_multiple_kpoints(self):
        Ebands = np.array([[0.0, 1.0], [2.0, 4.0]])
        result = one_over_E_diff(Ebands)
        assert result[0, 0, 1] == pytest.approx(-1.0)  # 1/(0-1)
        assert result[1, 0, 1] == pytest.approx(-0.5)  # 1/(2-4)


# ─────────────────────────────────────────────────────────────
# apply_Qshift_on_valence_states
# ─────────────────────────────────────────────────────────────

class TestApplyQshiftOnValenceStates:
    def test_zero_qshift_returns_gv_unchanged(self):
        Qshift = np.array([0.0, 0.0, 0.0])
        Nk, Nm, Nc, Nv = 3, 2, 2, 2
        Gv   = np.random.rand(Nm, Nk, Nc, Nv, Nc, Nv)
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        result = apply_Qshift_on_valence_states(Qshift, Gv, kpts, verbose=False)
        assert np.allclose(result, Gv)

    def test_nonzero_qshift_permutes_k_axis(self):
        # 2-point 1D grid: k = [0, 0.5]. Shift by 0.5 → shifted = [0.5, 1.0≡0.0]
        # mapping: k0(0.0) → index of 0.5 = 1; k1(0.5) → index of 0.0 = 0
        Qshift = np.array([0.5, 0.0, 0.0])
        Nk = 2
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        rng = np.random.default_rng(0)
        Gv = rng.random((1, Nk, 1, 1, 1, 1))
        result = apply_Qshift_on_valence_states(Qshift, Gv, kpts, verbose=False)
        assert np.allclose(result[:, 0, ...], Gv[:, 1, ...])
        assert np.allclose(result[:, 1, ...], Gv[:, 0, ...])

    def test_identity_qshift_for_uniform_grid(self):
        # Shift by a full G vector should map every k back to itself
        Qshift = np.array([1.0, 0.0, 0.0])  # full reciprocal lattice vector
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        Nk = 2
        Gv = np.arange(float(Nk)).reshape(1, Nk, 1, 1, 1, 1)
        result = apply_Qshift_on_valence_states(Qshift, Gv, kpts, verbose=False)
        assert np.allclose(result, Gv)


# ─────────────────────────────────────────────────────────────
# read_exciton_pairs
# ─────────────────────────────────────────────────────────────

class TestReadExcitonPairs:
    def test_no_file_uses_iexc_jexc(self):
        cfg = {'read_exciton_pairs_file': False, 'iexc': 1, 'jexc': 2, 'exciton_pairs': []}
        read_exciton_pairs(cfg)
        assert cfg['exciton_pairs'] == [(1, 2)]

    def test_single_column_file_creates_diagonal_pairs(self, tmp_path):
        f = tmp_path / 'exciton_pairs.dat'
        f.write_text('1\n2\n3\n')
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = {'read_exciton_pairs_file': True, 'iexc': 1, 'jexc': 1, 'exciton_pairs': []}
            read_exciton_pairs(cfg)
            assert cfg['exciton_pairs'] == [(1, 1), (2, 2), (3, 3)]
        finally:
            os.chdir(cwd)

    def test_two_column_file(self, tmp_path):
        f = tmp_path / 'exciton_pairs.dat'
        f.write_text('1 2\n3 4\n5 5\n')
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = {'read_exciton_pairs_file': True, 'iexc': 1, 'jexc': 1, 'exciton_pairs': []}
            read_exciton_pairs(cfg)
            assert cfg['exciton_pairs'] == [(1, 2), (3, 4), (5, 5)]
        finally:
            os.chdir(cwd)

    def test_missing_file_falls_back_to_config(self, tmp_path):
        cwd = os.getcwd()
        os.chdir(tmp_path)  # no exciton_pairs.dat here
        try:
            cfg = {'read_exciton_pairs_file': True, 'iexc': 2, 'jexc': 3, 'exciton_pairs': []}
            read_exciton_pairs(cfg)
            assert cfg['exciton_pairs'] == [(2, 3)]
        finally:
            os.chdir(cwd)

    def test_diagonal_when_iexc_equals_jexc(self):
        cfg = {'read_exciton_pairs_file': False, 'iexc': 4, 'jexc': 4, 'exciton_pairs': []}
        read_exciton_pairs(cfg)
        assert cfg['exciton_pairs'] == [(4, 4)]


# ─────────────────────────────────────────────────────────────
# Vectorized vs non-vectorized consistency
# ─────────────────────────────────────────────────────────────

def _build_dRPA_mat(elph_cond, elph_val):
    """Expand elph → full (Nmodes, Nk, Nc, Nv, Nc, Nv) dRPA matrix."""
    Nmodes, Nk, Nc, _ = elph_cond.shape
    Nv = elph_val.shape[2]
    Gc = np.zeros((Nmodes, Nk, Nc, Nv, Nc, Nv), dtype=complex)
    Gv = np.zeros_like(Gc)
    for iv in range(Nv):
        Gc[:, :, :, iv, :, iv] = elph_cond
    for ic in range(Nc):
        Gv[:, :, ic, :, ic, :] = elph_val
    return Gc - Gv


def _build_dRPA_diag_mat(elph_cond, elph_val):
    """Build diagonal (Nmodes, Nk, Nc, Nv) dRPA matrix."""
    Gc_diag = np.diagonal(elph_cond, axis1=2, axis2=3)[:, :, :, np.newaxis]
    Gv_diag = np.diagonal(elph_val,  axis1=2, axis2=3)[:, :, np.newaxis, :]
    return Gc_diag - Gv_diag


class TestVectorizedConsistency:
    """Vectorized and non-vectorized implementations must agree on small toy systems."""

    def _toy_system(self, Nk=2, Nc=2, Nv=2, Nmodes=3, seed=42):
        rng = np.random.default_rng(seed)
        Akcv = rng.random((Nk, Nc, Nv)) + 1j * rng.random((Nk, Nc, Nv))
        Bkcv = rng.random((Nk, Nc, Nv)) + 1j * rng.random((Nk, Nc, Nv))
        elph_cond = rng.random((Nmodes, Nk, Nc, Nc)) + 1j * rng.random((Nmodes, Nk, Nc, Nc))
        elph_val  = rng.random((Nmodes, Nk, Nv, Nv)) + 1j * rng.random((Nmodes, Nk, Nv, Nv))
        return Akcv, Bkcv, elph_cond, elph_val

    def test_dRPA_vectorized_vs_not_vectorized(self):
        Nk, Nc, Nv, Nm = 2, 3, 2, 4
        Akcv, Bkcv, ec, ev = self._toy_system(Nk, Nc, Nv, Nm)
        dRPA = _build_dRPA_mat(ec, ev)
        f_vec  = compute_A_dRPA_dr_imode_B_vectorized(Akcv, Bkcv, dRPA)
        f_nvec = compute_A_dRPA_dr_imode_B_not_vectorized(Akcv, Bkcv, ec, ev)
        assert np.allclose(f_vec, f_nvec, atol=1e-10), \
            f"Max diff: {np.max(np.abs(f_vec - f_nvec))}"

    def test_dRPAdiag_vectorized_vs_not_vectorized(self):
        Nk, Nc, Nv, Nm = 2, 3, 2, 4
        Akcv, Bkcv, ec, ev = self._toy_system(Nk, Nc, Nv, Nm)
        dRPA_diag = _build_dRPA_diag_mat(ec, ev)
        f_vec  = compute_A_dRPAdiag_dr_imode_B_vectorized(Akcv, Bkcv, dRPA_diag)
        f_nvec = compute_A_dRPAdiag_dr_imode_B_not_vectorized(Akcv, Bkcv, ec, ev)
        assert np.allclose(f_vec, f_nvec, atol=1e-10), \
            f"Max diff: {np.max(np.abs(f_vec - f_nvec))}"

    def test_dKernel_matrix_vectorized_vs_not_vectorized(self):
        Nk, Nc, Nv, Nm = 2, 2, 2, 2
        rng = np.random.default_rng(99)
        kernel = rng.random((Nk, Nc, Nv, Nk, Nc, Nv)) + 1j * rng.random((Nk, Nc, Nv, Nk, Nc, Nv))
        ec = rng.random((Nm, Nk, Nc, Nc)) + 1j * rng.random((Nm, Nk, Nc, Nc))
        ev = rng.random((Nm, Nk, Nv, Nv)) + 1j * rng.random((Nm, Nk, Nv, Nv))
        # Non-degenerate energies to avoid divide-by-zero in non-vectorized
        Econd = np.array([[1.0, 3.0], [5.0, 7.0]])
        Eval  = np.array([[0.0, 0.4], [0.8, 1.2]])
        for imode in range(Nm):
            r_vec  = calc_Dkernel_vectorized(kernel, ec, ev, Econd, Eval, imode)
            r_nvec = calc_Dkernel_single_mode_not_vectorized(kernel, ec, ev, Econd, Eval, imode)
            assert np.allclose(r_vec, r_nvec, atol=1e-10), \
                f"Mismatch for imode={imode}, max diff: {np.max(np.abs(r_vec - r_nvec))}"

    def test_dKernel_contraction_vectorized_vs_not_vectorized(self):
        Nk, Nc, Nv, Nm = 2, 2, 2, 3
        rng = np.random.default_rng(7)
        Akcv = rng.random((Nk, Nc, Nv)) + 1j * rng.random((Nk, Nc, Nv))
        Bkcv = rng.random((Nk, Nc, Nv)) + 1j * rng.random((Nk, Nc, Nv))
        DK = rng.random((Nm, Nk, Nc, Nv, Nk, Nc, Nv)) + 1j * rng.random((Nm, Nk, Nc, Nv, Nk, Nc, Nv))
        f_vec  = compute_A_dKernel_dr_imode_B_vectorized(Akcv, Bkcv, DK)
        f_nvec = compute_A_dKernel_dr_imode_B_not_vectorized(Akcv, Bkcv, DK)
        assert np.allclose(f_vec, f_nvec, atol=1e-10), \
            f"Max diff: {np.max(np.abs(f_vec - f_nvec))}"

    def test_dRPA_and_dRPAdiag_match_when_diagonal(self):
        # When elph_cond and elph_val are diagonal (off-diag = 0),
        # dRPA and dRPAdiag should give the same forces.
        Nk, Nc, Nv, Nm = 2, 2, 2, 3
        rng = np.random.default_rng(5)
        Akcv = rng.random((Nk, Nc, Nv)) + 1j * rng.random((Nk, Nc, Nv))
        # Force A = B (diagonal case: forces are real)
        Bkcv = Akcv.copy()
        ec = np.zeros((Nm, Nk, Nc, Nc), dtype=complex)
        ev = np.zeros((Nm, Nk, Nv, Nv), dtype=complex)
        for imode in range(Nm):
            for ik in range(Nk):
                ec[imode, ik] = np.diag(rng.random(Nc))
                ev[imode, ik] = np.diag(rng.random(Nv))

        dRPA      = _build_dRPA_mat(ec, ev)
        dRPA_diag = _build_dRPA_diag_mat(ec, ev)

        f_rpa  = compute_A_dRPA_dr_imode_B_vectorized(Akcv, Bkcv, dRPA)
        f_diag = compute_A_dRPAdiag_dr_imode_B_vectorized(Akcv, Bkcv, dRPA_diag)
        assert np.allclose(f_rpa, f_diag, atol=1e-10), \
            f"dRPA and dRPAdiag differ for diagonal elph: max diff {np.max(np.abs(f_rpa - f_diag))}"
