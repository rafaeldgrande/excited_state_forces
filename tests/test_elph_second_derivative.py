"""
Tests for elph/elph_coeffs_second_derivative.py:
  _inv_dE, compute_g2_cart, _build_q_map, read_eqp
"""
import pytest
import numpy as np
from elph_coeffs_second_derivative import _inv_dE, compute_g2_cart, _build_q_map, read_eqp


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
