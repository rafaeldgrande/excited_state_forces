"""
Tests for elph/elph_coeffs_second_derivative.py:
  _inv_dE, compute_g2_cart, _build_q_map, read_eqp, read_eqp_full_range
"""
import h5py
import pytest
import numpy as np
from elph_coeffs_second_derivative import (
    _inv_dE, compute_g2_cart, _build_q_map, read_eqp, read_eqp_full_range,
)
from elph_xml_to_h5_QE import RY_TO_EV, build_qp_rescaling_ratio


# ─────────────────────────────────────────────────────────────
# _inv_dE
# ─────────────────────────────────────────────────────────────

class TestInvDE:
    def test_output_shape(self):
        E = np.ones((3, 4))
        result = _inv_dE(E)
        assert result.shape == (3, 4, 4)

    def test_diagonal_is_zero(self):
        E = np.array([[1.0, 2.0, 3.0]])
        result = _inv_dE(E)
        for ib in range(3):
            assert result[0, ib, ib] == pytest.approx(0.0)

    def test_off_diagonal_values(self):
        E = np.array([[1.0, 2.0]])
        result = _inv_dE(E)
        assert result[0, 0, 1] == pytest.approx(1.0 / (1.0 - 2.0))
        assert result[0, 1, 0] == pytest.approx(1.0 / (2.0 - 1.0))

    def test_antisymmetry(self):
        E = np.arange(6, dtype=float).reshape(2, 3) + 0.5
        result = _inv_dE(E)
        assert np.allclose(result, -result.transpose(0, 2, 1))

    def test_degenerate_bands_give_zero(self):
        E = np.array([[2.0, 2.0]])
        result = _inv_dE(E)
        assert np.allclose(result, 0.0)

    def test_multiple_kpoints(self):
        E = np.array([[0.0, 1.0], [2.0, 4.0]])
        result = _inv_dE(E)
        assert result[0, 0, 1] == pytest.approx(-1.0)   # 1/(0-1)
        assert result[1, 0, 1] == pytest.approx(-0.5)   # 1/(2-4)


# ─────────────────────────────────────────────────────────────
# compute_g2_cart
# ─────────────────────────────────────────────────────────────

class TestComputeG2Cart:
    def test_output_shape(self):
        Npert, Nk, Nb = 6, 3, 4
        g = np.random.rand(Npert, Nk, Nb, Nb) + 1j * np.random.rand(Npert, Nk, Nb, Nb)
        E = np.arange(Nk * Nb, dtype=float).reshape(Nk, Nb)
        inv_dE = _inv_dE(E)
        g2 = compute_g2_cart(g, inv_dE)
        assert g2.shape == (Npert, Nk, Nb, Nb)

    def test_zero_when_inv_dE_is_zero(self):
        # All energies equal → inv_dE = 0 everywhere → g2 = 0
        Npert, Nk, Nb = 2, 2, 3
        g = np.random.rand(Npert, Nk, Nb, Nb) + 1j * np.random.rand(Npert, Nk, Nb, Nb)
        inv_dE = np.zeros((Nk, Nb, Nb))
        g2 = compute_g2_cart(g, inv_dE)
        assert np.allclose(g2, 0.0)

    def test_zero_when_g_is_identity(self):
        # Diagonal g (identity) has no off-diagonal coupling → g2 = 0
        Npert, Nk, Nb = 1, 1, 2
        g = np.zeros((Npert, Nk, Nb, Nb), dtype=complex)
        g[0, 0] = np.eye(Nb)
        E = np.array([[1.0, 3.0]])
        inv_dE = _inv_dE(E)
        g2 = compute_g2_cart(g, inv_dE)
        assert np.allclose(g2, 0.0)

    def test_analytical_2band_case(self):
        # For Npert=1, Nk=1, Nb=2, g=[[1,2],[3,4]], E=[1,2]:
        # g2 = [[2bc, b(d-a)], [c(d-a), -2bc]] = [[12, 6], [9, -12]]
        # where a,b,c,d = 1,2,3,4
        g = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex).reshape(1, 1, 2, 2)
        E = np.array([[1.0, 2.0]])
        inv_dE = _inv_dE(E)
        g2 = compute_g2_cart(g, inv_dE)
        expected = np.array([[12.0, 6.0], [9.0, -12.0]], dtype=complex).reshape(1, 1, 2, 2)
        assert np.allclose(g2, expected, atol=1e-10)

    def test_term1_plus_term2_symmetry(self):
        # g2 should be consistent: for hermitian g and real E,
        # g2[n,m] = conj(g2[m,n]) (since g2 is hermitian when g is hermitian)
        Npert, Nk, Nb = 2, 2, 3
        rng = np.random.default_rng(7)
        # Make hermitian g
        g_raw = rng.random((Npert, Nk, Nb, Nb)) + 1j * rng.random((Npert, Nk, Nb, Nb))
        g = g_raw + g_raw.conj().transpose(0, 1, 3, 2)
        E = np.sort(rng.random((Nk, Nb)) * 5.0, axis=1)
        inv_dE = _inv_dE(E)
        g2 = compute_g2_cart(g, inv_dE)
        # g2 should be hermitian: g2[a,k,n,m] = conj(g2[a,k,m,n])
        assert np.allclose(g2, g2.conj().transpose(0, 1, 3, 2), atol=1e-10)

    def test_naive_vs_vectorized(self):
        # Cross-check compute_g2_cart against a plain Python loop implementation
        Npert, Nk, Nb = 3, 2, 3
        rng = np.random.default_rng(42)
        g = rng.random((Npert, Nk, Nb, Nb)) + 1j * rng.random((Npert, Nk, Nb, Nb))
        E = np.arange(Nk * Nb, dtype=float).reshape(Nk, Nb) * 0.5 + 0.1
        inv_dE = _inv_dE(E)
        g2_fast = compute_g2_cart(g, inv_dE)

        # Naive loop
        g2_naive = np.zeros_like(g)
        for a in range(Npert):
            for k in range(Nk):
                for n in range(Nb):
                    for m in range(Nb):
                        s = 0.0 + 0j
                        for l in range(Nb):
                            fac = inv_dE[k, n, l] - inv_dE[k, l, m]
                            s -= g[a, k, n, l] * g[a, k, l, m] * fac
                        g2_naive[a, k, n, m] = s

        assert np.allclose(g2_fast, g2_naive, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# _build_q_map
# ─────────────────────────────────────────────────────────────

class TestBuildQMap:
    def test_exact_matches(self):
        qph = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        result = _build_q_map(qph, qph)
        assert list(result) == [0, 1, 2]

    def test_not_found_gives_minus_one(self):
        elph = np.array([[0.3, 0.0, 0.0]])
        ph   = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        result = _build_q_map(elph, ph)
        assert result[0] == -1

    def test_partial_match(self):
        elph = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
        ph   = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        result = _build_q_map(elph, ph)
        assert result[0] == 0
        assert result[1] == -1

    def test_returns_first_match(self):
        # Two identical ph q-points — should map to the first one (index 0)
        elph = np.array([[0.5, 0.0, 0.0]])
        ph   = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        result = _build_q_map(elph, ph)
        assert result[0] == 0

    def test_tolerance_just_inside(self):
        elph = np.array([[0.0, 0.0, 0.0]])
        ph   = np.array([[0.5e-6, 0.0, 0.0]])   # well within default tol=1e-5
        result = _build_q_map(elph, ph)
        assert result[0] == 0

    def test_tolerance_just_outside(self):
        elph = np.array([[0.0, 0.0, 0.0]])
        ph   = np.array([[2e-5, 0.0, 0.0]])      # outside default tol=1e-5
        result = _build_q_map(elph, ph)
        assert result[0] == -1


# ─────────────────────────────────────────────────────────────
# read_eqp (elph_coeffs_second_derivative version)
# ─────────────────────────────────────────────────────────────

class TestReadEqpElph:
    def _make_eqp_file(self, tmp_path, Nk, Nc, Nv, Nval):
        """Minimal eqp1.dat with a QP correction that grows with ik."""
        lines = []
        for ik in range(Nk):
            lines.append(f'  0.000 0.000 {ik * 0.1:.3f}')
            # valence: band indices Nval down to Nval-Nv+1
            for iv_file in range(Nval, Nval - Nv, -1):
                dft = iv_file * 0.2
                qp  = dft + 0.05 * ik
                lines.append(f'1 {iv_file} {dft:.6f} {qp:.6f}')
            # conduction: band indices Nval+1 to Nval+Nc
            for ic_file in range(Nval + 1, Nval + Nc + 1):
                dft = ic_file * 0.2
                qp  = dft + 0.10 * ik
                lines.append(f'1 {ic_file} {dft:.6f} {qp:.6f}')
        f = tmp_path / 'eqp1.dat'
        f.write_text('\n'.join(lines) + '\n')
        return str(f)

    def test_output_shapes(self, tmp_path):
        Nk, Nc, Nv, Nval = 3, 4, 2, 10
        f = self._make_eqp_file(tmp_path, Nk, Nc, Nv, Nval)
        Eqp_c, Eqp_v, Edft_c, Edft_v = read_eqp(f, Nk, Nc, Nv, Nval)
        assert Eqp_c.shape  == (Nk, Nc)
        assert Eqp_v.shape  == (Nk, Nv)
        assert Edft_c.shape == (Nk, Nc)
        assert Edft_v.shape == (Nk, Nv)

    def test_at_first_kpoint_qp_equals_dft(self, tmp_path):
        # ik=0 has zero correction (0.05*0 = 0)
        Nk, Nc, Nv, Nval = 2, 2, 2, 6
        f = self._make_eqp_file(tmp_path, Nk, Nc, Nv, Nval)
        Eqp_c, Eqp_v, Edft_c, Edft_v = read_eqp(f, Nk, Nc, Nv, Nval)
        assert np.allclose(Eqp_c[0], Edft_c[0])
        assert np.allclose(Eqp_v[0], Edft_v[0])

    def test_homo_is_first_valence_band(self, tmp_path):
        # iv=0 should correspond to file band Nval (HOMO): ic=0=iv=0 → band ibnd=Nval
        Nk, Nc, Nv, Nval = 1, 1, 2, 4
        f = self._make_eqp_file(tmp_path, Nk, Nc, Nv, Nval)
        Eqp_c, Eqp_v, Edft_c, Edft_v = read_eqp(f, Nk, Nc, Nv, Nval)
        # iv=0 ↔ ibnd=Nval=4: dft = 4 * 0.2 = 0.8
        assert Edft_v[0, 0] == pytest.approx(0.8)
        # iv=1 ↔ ibnd=Nval-1=3: dft = 3 * 0.2 = 0.6
        assert Edft_v[0, 1] == pytest.approx(0.6)

    def test_lumo_is_first_conduction_band(self, tmp_path):
        # ic=0 ↔ ibnd=Nval+1 (LUMO)
        Nk, Nc, Nv, Nval = 1, 2, 1, 4
        f = self._make_eqp_file(tmp_path, Nk, Nc, Nv, Nval)
        Eqp_c, Eqp_v, Edft_c, Edft_v = read_eqp(f, Nk, Nc, Nv, Nval)
        # ic=0 ↔ ibnd=Nval+1=5: dft = 5 * 0.2 = 1.0
        assert Edft_c[0, 0] == pytest.approx(1.0)
        # ic=1 ↔ ibnd=Nval+2=6: dft = 6 * 0.2 = 1.2
        assert Edft_c[0, 1] == pytest.approx(1.2)


# ─────────────────────────────────────────────────────────────
# read_eqp_full_range
# ─────────────────────────────────────────────────────────────

class TestReadEqpFullRange:
    def _wfn_h5(self, tmp_path, kpts_crystal, eigenvalues_ev, Nval, nat=1):
        """Minimal WFN.h5 with just the mf_header fields read_wfn_h5_header needs."""
        path = tmp_path / 'WFN_test.h5'
        nrk = len(kpts_crystal)
        mnband = eigenvalues_ev.shape[1]
        with h5py.File(path, 'w') as fh:
            mf = fh.create_group('mf_header')
            kp = mf.create_group('kpoints')
            kp.create_dataset('rk', data=np.asarray(kpts_crystal, dtype=np.float64))
            kp.create_dataset('mnband', data=mnband)
            kp.create_dataset('nspin', data=1)
            kp.create_dataset('ifmax', data=np.full((1, nrk), Nval, dtype=np.int32))
            el_ry = np.asarray(eigenvalues_ev, dtype=np.float64) / RY_TO_EV
            kp.create_dataset('el', data=el_ry[None, :, :])   # (nspin, nrk, mnband)
            cr = mf.create_group('crystal')
            cr.create_dataset('alat', data=1.0)
            cr.create_dataset('avec', data=np.eye(3))
            cr.create_dataset('bvec', data=np.eye(3))
            cr.create_dataset('nat', data=nat)
            cr.create_dataset('atyp', data=np.ones(nat, dtype=np.int32))
            cr.create_dataset('apos', data=np.zeros((nat, 3)))
        return str(path)

    def test_no_fallback_matches_plain_read_eqp(self, tmp_path):
        # Nc/Nv already match eqp.dat's own window -- no WFN.h5 needed at all.
        Nk, Nc, Nv, Nval = 2, 2, 2, 6
        f = TestReadEqpElph()._make_eqp_file(tmp_path, Nk, Nc, Nv, Nval)
        Eqp_c1, Eqp_v1, Edft_c1, Edft_v1 = read_eqp(f, Nk, Nc, Nv, Nval)
        Eqp_c2, Eqp_v2, Edft_c2, Edft_v2, ncw, nvw = read_eqp_full_range(
            f, Nk, Nc, Nv, Nval, kpoints_crystal=np.zeros((Nk, 3)), wfn_dfpt_path=None)
        assert (ncw, nvw) == (Nc, Nv)
        assert np.allclose(Eqp_c1, Eqp_c2)
        assert np.allclose(Eqp_v1, Eqp_v2)
        assert np.allclose(Edft_c1, Edft_c2)
        assert np.allclose(Edft_v1, Edft_v2)

    def test_missing_wfn_dfpt_raises(self, tmp_path):
        Nk, Nc, Nv, Nval = 1, 2, 2, 6
        f = TestReadEqpElph()._make_eqp_file(tmp_path, Nk, Nc, Nv, Nval)
        with pytest.raises(ValueError, match='wfn_dfpt'):
            read_eqp_full_range(f, Nk, Nc + 2, Nv, Nval,
                                 kpoints_crystal=np.zeros((Nk, 3)), wfn_dfpt_path=None)

    def test_out_of_window_uses_dft_fallback(self, tmp_path):
        Nk, Nc_window, Nv_window, Nval = 1, 2, 2, 6
        f = TestReadEqpElph()._make_eqp_file(tmp_path, Nk, Nc_window, Nv_window, Nval)
        Nc_avail, Nv_avail = Nc_window + 2, Nv_window + 1
        kpts = np.array([[0.0, 0.0, 0.0]])
        mnband = Nval + Nc_avail
        eig = np.arange(mnband, dtype=float).reshape(1, mnband) * 0.11  # eV, arbitrary
        wfn = self._wfn_h5(tmp_path, kpts, eig, Nval)

        Eqp_c, Eqp_v, Edft_c, Edft_v, ncw, nvw = read_eqp_full_range(
            f, Nk, Nc_avail, Nv_avail, Nval, kpoints_crystal=kpts, wfn_dfpt_path=wfn)
        assert (ncw, nvw) == (Nc_window, Nv_window)

        # In-window bands: unaffected by the fallback, still eqp.dat's QP energies.
        Eqp_c_win, Eqp_v_win, _, _ = read_eqp(f, Nk, Nc_window, Nv_window, Nval)
        assert np.allclose(Eqp_c[:, :Nc_window], Eqp_c_win)
        assert np.allclose(Eqp_v[:, :Nv_window], Eqp_v_win)

        # Out-of-window bands: filled from the WFN.h5 DFT eigenvalues, not left at 0.
        for ic in range(Nc_window, Nc_avail):
            assert Eqp_c[0, ic] == pytest.approx(eig[0, Nval + ic])
        for iv in range(Nv_window, Nv_avail):
            assert Eqp_v[0, iv] == pytest.approx(eig[0, Nval - 1 - iv])

    def test_out_of_window_edft_equals_eqp_fallback(self, tmp_path):
        # The out-of-window fallback sets Edft equal to Eqp (both = the DFT
        # eigenvalue) -- this is what makes build_qp_rescaling_ratio come out
        # to 1.0 for band pairs entirely in the fallback region (see below).
        Nk, Nc_window, Nv_window, Nval = 1, 1, 1, 5
        f = TestReadEqpElph()._make_eqp_file(tmp_path, Nk, Nc_window, Nv_window, Nval)
        Nc_avail, Nv_avail = Nc_window + 2, Nv_window + 2
        kpts = np.array([[0.0, 0.0, 0.0]])
        mnband = Nval + Nc_avail
        eig = np.arange(mnband, dtype=float).reshape(1, mnband) * 0.13
        wfn = self._wfn_h5(tmp_path, kpts, eig, Nval)

        Eqp_c, Eqp_v, Edft_c, Edft_v, _, _ = read_eqp_full_range(
            f, Nk, Nc_avail, Nv_avail, Nval, kpoints_crystal=kpts, wfn_dfpt_path=wfn)
        assert np.allclose(Eqp_c[:, Nc_window:], Edft_c[:, Nc_window:])
        assert np.allclose(Eqp_v[:, Nv_window:], Edft_v[:, Nv_window:])


# ─────────────────────────────────────────────────────────────
# build_qp_rescaling_ratio (imported from elph_xml_to_h5)
# ─────────────────────────────────────────────────────────────

class TestBuildQpRescalingRatio:
    def test_diagonal_is_one(self):
        Eqp  = np.array([[1.0, 2.5, 4.0]])
        Edft = np.array([[0.8, 2.0, 3.5]])
        ratio = build_qp_rescaling_ratio(Eqp, Edft)
        for ib in range(3):
            assert ratio[0, ib, ib] == pytest.approx(1.0)

    def test_degenerate_dft_pair_gives_one(self):
        # Edft difference below tol_deg -> ratio=1.0 even if Eqp differs
        Eqp  = np.array([[1.0, 3.0]])
        Edft = np.array([[2.0, 2.0]])
        ratio = build_qp_rescaling_ratio(Eqp, Edft, tol_deg=1e-5)
        assert np.allclose(ratio, 1.0)

    def test_analytical_ratio(self):
        Eqp  = np.array([[1.0, 3.0]])
        Edft = np.array([[0.5, 2.5]])
        ratio = build_qp_rescaling_ratio(Eqp, Edft)
        expected = (1.0 - 3.0) / (0.5 - 2.5)
        assert ratio[0, 0, 1] == pytest.approx(expected)
        assert ratio[0, 1, 0] == pytest.approx(expected)   # symmetric: both diffs flip sign

    def test_eqp_equals_edft_gives_all_ones(self):
        # No QP correction anywhere -> ratio is identically 1 -> raw DFT el-ph unchanged
        E = np.array([[0.1, 1.2, 2.3, 3.4]])
        ratio = build_qp_rescaling_ratio(E, E)
        assert np.allclose(ratio, 1.0)

    def test_multiple_kpoints_independent(self):
        Eqp  = np.array([[1.0, 3.0], [0.0, 10.0]])
        Edft = np.array([[0.5, 2.5], [0.0, 5.0]])
        ratio = build_qp_rescaling_ratio(Eqp, Edft)
        assert ratio[0, 0, 1] == pytest.approx((1.0 - 3.0) / (0.5 - 2.5))
        assert ratio[1, 0, 1] == pytest.approx((0.0 - 10.0) / (0.0 - 5.0))
