"""Tests for bgw_interface_m.py: reverse_bse_index, rpa_part_from_eqp, read_eqp_data."""
import pytest
import numpy as np
from bgw_interface_m import reverse_bse_index, rpa_part_from_eqp, read_eqp_data
from excited_forces_classes import Parameters_BSE


class TestReverseBSEIndex:
    def test_ibse_zero_gives_all_zeros(self):
        ik, ic, iv = reverse_bse_index(0, 5, 4, 3)
        assert (ik, ic, iv) == (0, 0, 0)

    def test_v_is_innermost_index(self):
        # v cycles fastest: ibse=1 → (ik=0, ic=0, iv=1)
        ik, ic, iv = reverse_bse_index(1, 3, 4, 5)
        assert (ik, ic, iv) == (0, 0, 1)

    def test_c_index_increments_after_nv(self):
        Nk, Nc, Nv = 3, 4, 5
        ik, ic, iv = reverse_bse_index(Nv, Nk, Nc, Nv)
        assert (ik, ic, iv) == (0, 1, 0)

    def test_k_index_increments_after_nc_times_nv(self):
        Nk, Nc, Nv = 3, 4, 5
        ik, ic, iv = reverse_bse_index(Nc * Nv, Nk, Nc, Nv)
        assert (ik, ic, iv) == (1, 0, 0)

    def test_last_index(self):
        Nk, Nc, Nv = 3, 4, 5
        total = Nk * Nc * Nv - 1
        ik, ic, iv = reverse_bse_index(total, Nk, Nc, Nv)
        assert (ik, ic, iv) == (Nk - 1, Nc - 1, Nv - 1)

    def test_roundtrip_all_indices(self):
        Nk, Nc, Nv = 3, 4, 5
        for ibse in range(Nk * Nc * Nv):
            ik, ic, iv = reverse_bse_index(ibse, Nk, Nc, Nv)
            assert 0 <= ik < Nk
            assert 0 <= ic < Nc
            assert 0 <= iv < Nv
            ibse_back = ik * Nc * Nv + ic * Nv + iv
            assert ibse_back == ibse

    def test_nspin_1_is_default(self):
        ik1, ic1, iv1 = reverse_bse_index(7, 3, 4, 5, Nspin=1)
        ik2, ic2, iv2 = reverse_bse_index(7, 3, 4, 5)
        assert (ik1, ic1, iv1) == (ik2, ic2, iv2)


class TestRPAPartFromEQP:
    def test_shape(self):
        Eqp_cond = np.ones((3, 4))
        Eqp_val  = np.ones((3, 2))
        rpa = rpa_part_from_eqp(Eqp_cond, Eqp_val)
        assert rpa.shape == (3, 4, 2, 3, 4, 2)

    def test_diagonal_is_econd_minus_eval(self):
        Eqp_cond = np.array([[2.0, 3.0], [4.0, 5.0]])
        Eqp_val  = np.array([[0.0, 0.5], [1.0, 1.5]])
        rpa = rpa_part_from_eqp(Eqp_cond, Eqp_val)
        # rpa[ik, ic, iv, ik, ic, iv] = Econd[ik,ic] - Eval[ik,iv]
        assert rpa[0, 0, 0, 0, 0, 0] == pytest.approx(2.0 - 0.0)
        assert rpa[0, 0, 1, 0, 0, 1] == pytest.approx(2.0 - 0.5)
        assert rpa[0, 1, 0, 0, 1, 0] == pytest.approx(3.0 - 0.0)
        assert rpa[1, 0, 0, 1, 0, 0] == pytest.approx(4.0 - 1.0)
        assert rpa[1, 1, 1, 1, 1, 1] == pytest.approx(5.0 - 1.5)

    def test_off_diagonal_in_k_is_zero(self):
        Eqp_cond = np.array([[2.0, 3.0], [4.0, 5.0]])
        Eqp_val  = np.array([[0.0, 0.5], [1.0, 1.5]])
        rpa = rpa_part_from_eqp(Eqp_cond, Eqp_val)
        assert rpa[0, 0, 0, 1, 0, 0] == pytest.approx(0.0)

    def test_off_diagonal_in_bands_is_zero(self):
        Eqp_cond = np.array([[2.0, 3.0]])
        Eqp_val  = np.array([[0.0, 0.5]])
        rpa = rpa_part_from_eqp(Eqp_cond, Eqp_val)
        assert rpa[0, 0, 0, 0, 1, 0] == pytest.approx(0.0)
        assert rpa[0, 0, 0, 0, 0, 1] == pytest.approx(0.0)

    def test_only_diagonal_nonzero(self):
        Eqp_cond = np.array([[1.0, 2.0]])
        Eqp_val  = np.array([[0.0]])
        rpa = rpa_part_from_eqp(Eqp_cond, Eqp_val)
        # off-diagonal sum should be exactly zero
        diag_sum = rpa[0, 0, 0, 0, 0, 0] + rpa[0, 1, 0, 0, 1, 0]
        total_sum = rpa.sum()
        assert total_sum == pytest.approx(diag_sum)


class TestReadEQPData:
    """Tests that read_eqp_data correctly parses a minimal eqp.dat file."""

    def _make_eqp_file(self, tmp_path, Nkpoints, Nvbnds, Ncbnds, Nval):
        lines = []
        for ik in range(Nkpoints):
            lines.append(f'  0.000 0.000 {ik * 0.1:.3f}')  # k-point header (first entry != '1')
            # valence bands: file indices Nval down to Nval-Nvbnds+1
            for iv_file in range(Nval, Nval - Nvbnds, -1):
                dft = iv_file * 0.1
                qp  = dft + 0.01 * ik
                lines.append(f'1 {iv_file} {dft:.6f} {qp:.6f}')
            # conduction bands: file indices Nval+1 to Nval+Ncbnds
            for ic_file in range(Nval + 1, Nval + Ncbnds + 1):
                dft = ic_file * 0.1
                qp  = dft + 0.02 * ik
                lines.append(f'1 {ic_file} {dft:.6f} {qp:.6f}')
        f = tmp_path / 'eqp1.dat'
        f.write_text('\n'.join(lines) + '\n')
        return str(f)

    def _bse_params(self, Nk, Nc, Nv, Nval):
        kpts = np.zeros((Nk, 3))
        return Parameters_BSE(Nk, kpts, Nc, Nv, Nval, Nc, Nv, np.eye(3))

    def test_output_shapes(self, tmp_path):
        Nk, Nv, Nc, Nval = 2, 3, 2, 8
        eqp_file = self._make_eqp_file(tmp_path, Nk, Nv, Nc, Nval)
        bse = self._bse_params(Nk, Nc, Nv, Nval)
        Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, bse)
        assert Eqp_val.shape  == (Nk, Nv)
        assert Eqp_cond.shape == (Nk, Nc)
        assert Edft_val.shape  == (Nk, Nv)
        assert Edft_cond.shape == (Nk, Nc)

    def test_single_kpoint(self, tmp_path):
        Nk, Nv, Nc, Nval = 1, 2, 2, 4
        eqp_file = self._make_eqp_file(tmp_path, Nk, Nv, Nc, Nval)
        bse = self._bse_params(Nk, Nc, Nv, Nval)
        Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, bse)
        # For ik=0, correction is 0, so QP == DFT
        assert np.allclose(Eqp_val,  Edft_val)
        assert np.allclose(Eqp_cond, Edft_cond)

    def test_qp_differs_from_dft_at_second_kpoint(self, tmp_path):
        Nk, Nv, Nc, Nval = 2, 2, 2, 4
        eqp_file = self._make_eqp_file(tmp_path, Nk, Nv, Nc, Nval)
        bse = self._bse_params(Nk, Nc, Nv, Nval)
        Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, bse)
        # At ik=1 the QP correction is nonzero
        assert not np.allclose(Eqp_cond[1], Edft_cond[1])
