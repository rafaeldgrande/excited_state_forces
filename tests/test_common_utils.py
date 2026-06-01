"""
Tests for common/utils.py:
  _gb, _downsample_idx, unpolarized_invariant, read_eqp_dat_file
"""
import pytest
import numpy as np
from common import _gb, _downsample_idx, unpolarized_invariant, read_eqp_dat_file


# ─────────────────────────────────────────────────────────────
# _gb  (memory estimator)
# ─────────────────────────────────────────────────────────────

class TestGb:
    def test_single_float64_array(self):
        # 1 element × 8 bytes / 1024^3 bytes/GB
        result = _gb((1,), np.float64)
        assert result == pytest.approx(8 / 1024**3)

    def test_single_complex128_array(self):
        result = _gb((1,), np.complex128)
        assert result == pytest.approx(16 / 1024**3)

    def test_two_arrays_summed(self):
        a = _gb((100,), np.float64)
        b = _gb((200,), np.float64)
        ab = _gb((100,), np.float64, (200,), np.float64)
        assert ab == pytest.approx(a + b)

    def test_multidimensional_shape(self):
        result = _gb((10, 20, 30), np.float64)
        expected = 10 * 20 * 30 * 8 / 1024**3
        assert result == pytest.approx(expected)

    def test_exactly_1gb(self):
        # 1024^3 / 8 float64 elements = 1 GB
        n = 1024**3 // 8
        assert _gb((n,), np.float64) == pytest.approx(1.0)

    def test_no_arrays_returns_zero(self):
        assert _gb() == pytest.approx(0.0)

    def test_int32_itemsize(self):
        result = _gb((1,), np.int32)
        assert result == pytest.approx(4 / 1024**3)


# ─────────────────────────────────────────────────────────────
# _downsample_idx
# ─────────────────────────────────────────────────────────────

class TestDownsampleIdx:
    def test_identity_same_size(self):
        idx = _downsample_idx(5, 5)
        assert list(idx) == [0, 1, 2, 3, 4]

    def test_first_and_last_always_included(self):
        for n in [10, 20, 100]:
            idx = _downsample_idx(n, 3)
            assert idx[0] == 0
            assert idx[-1] == n - 1

    def test_single_point_returns_zero(self):
        idx = _downsample_idx(100, 1)
        assert list(idx) == [0]

    def test_two_points_returns_endpoints(self):
        idx = _downsample_idx(100, 2)
        assert list(idx) == [0, 99]

    def test_output_length_matches_target(self):
        for n_target in [3, 5, 10]:
            idx = _downsample_idx(50, n_target)
            assert len(idx) == n_target

    def test_output_is_integer_array(self):
        idx = _downsample_idx(10, 5)
        assert idx.dtype == np.int64 or np.issubdtype(idx.dtype, np.integer)

    def test_indices_are_strictly_increasing(self):
        idx = _downsample_idx(100, 10)
        assert np.all(np.diff(idx) >= 0)

    def test_all_indices_in_valid_range(self):
        n_full, n_target = 50, 7
        idx = _downsample_idx(n_full, n_target)
        assert np.all(idx >= 0)
        assert np.all(idx < n_full)


# ─────────────────────────────────────────────────────────────
# unpolarized_invariant
# ─────────────────────────────────────────────────────────────

class TestUnpolarizedInvariant:
    def test_zero_tensor(self):
        a = np.zeros((3, 3))
        assert unpolarized_invariant(a) == pytest.approx(0.0)

    def test_identity_tensor(self):
        # a = I: ᾱ=1, γ²=0, δ²=0 → 45*1² = 45
        a = np.eye(3)
        assert unpolarized_invariant(a) == pytest.approx(45.0)

    def test_scalar_multiple_of_identity(self):
        # a = c*I → 45|c|²
        c = 2.5
        a = c * np.eye(3)
        assert unpolarized_invariant(a) == pytest.approx(45 * c**2)

    def test_complex_scalar_identity(self):
        c = 1.0 + 1j
        a = c * np.eye(3, dtype=complex)
        assert unpolarized_invariant(a) == pytest.approx(45 * abs(c)**2)

    def test_antisymmetric_tensor(self):
        # Pure antisymmetric: a[0,1] = 1, a[1,0] = -1, others 0
        a = np.zeros((3, 3), dtype=complex)
        a[0, 1] = 1.0;  a[1, 0] = -1.0
        # ᾱ = 0, γ² = 3/4 * (|a[0,1]+a[1,0]|²+...) = 0
        # δ² = 3/4 * |a[0,1]-a[1,0]|² = 3/4 * 4 = 3
        result = unpolarized_invariant(a)
        assert result == pytest.approx(5.0 * 3.0)

    def test_symmetric_off_diagonal(self):
        # a[0,1] = a[1,0] = 1, others 0
        a = np.zeros((3, 3))
        a[0, 1] = 1.0;  a[1, 0] = 1.0
        # δ² = 0, γ² = 3/4 * 4 = 3 → result = 7*3 = 21
        assert unpolarized_invariant(a) == pytest.approx(21.0)

    def test_positive_semidefinite(self):
        # Invariant should always be >= 0 for any tensor
        rng = np.random.default_rng(42)
        a = rng.random((3, 3)) + 1j * rng.random((3, 3))
        assert unpolarized_invariant(a) >= 0.0

    def test_trailing_frequency_axis(self):
        # a has shape (3, 3, Nfreq) — verify result shape and values
        Nf = 5
        a = np.zeros((3, 3, Nf))
        for f in range(Nf):
            a[:, :, f] = (f + 1) * np.eye(3)   # c(f)*I → invariant = 45*c(f)²
        result = unpolarized_invariant(a)
        assert result.shape == (Nf,)
        for f in range(Nf):
            assert result[f] == pytest.approx(45.0 * (f + 1)**2)


# ─────────────────────────────────────────────────────────────
# read_eqp_dat_file
# ─────────────────────────────────────────────────────────────

def _write_eqp_dat(tmp_path, kpoints, dft_energies, qp_energies):
    """
    Create an eqp.dat-style file.
    kpoints: (Nk, 3), dft_energies/qp_energies: (Nbnds, Nk)
    """
    Nk = len(kpoints)
    Nbnds = dft_energies.shape[0]
    lines = []
    for ik in range(Nk):
        kx, ky, kz = kpoints[ik]
        lines.append(f'{kx:.6f} {ky:.6f} {kz:.6f} {Nbnds}')
        for ibnd in range(Nbnds):
            lines.append(f'1 {ibnd + 1} {dft_energies[ibnd, ik]:.6f} {qp_energies[ibnd, ik]:.6f}')
    path = tmp_path / 'eqp.dat'
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


class TestReadEqpDatFile:
    def test_output_shapes(self, tmp_path):
        Nk, Nbnds = 2, 3
        kpts = np.zeros((Nk, 3))
        dft  = np.arange(Nbnds * Nk, dtype=float).reshape(Nbnds, Nk)
        qp   = dft + 0.1
        f = _write_eqp_dat(tmp_path, kpts, dft, qp)
        bands_dft, bands_qp, Kpoints, nk, band_idx = read_eqp_dat_file(f)
        assert bands_dft.shape == (Nbnds, Nk)
        assert bands_qp.shape  == (Nbnds, Nk)
        assert Kpoints.shape   == (Nk, 3)
        assert nk == Nk

    def test_kpoints_parsed_correctly(self, tmp_path):
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        dft  = np.ones((2, 2))
        qp   = dft
        f = _write_eqp_dat(tmp_path, kpts, dft, qp)
        _, _, Kpoints, _, _ = read_eqp_dat_file(f)
        assert np.allclose(Kpoints[0], [0.0, 0.0, 0.0])
        assert np.allclose(Kpoints[1], [0.5, 0.0, 0.0])

    def test_energies_values(self, tmp_path):
        kpts = np.array([[0.0, 0.0, 0.0]])
        dft  = np.array([[1.0], [2.0]])   # (Nbnds=2, Nk=1)
        qp   = np.array([[1.1], [2.2]])
        f = _write_eqp_dat(tmp_path, kpts, dft, qp)
        bands_dft, bands_qp, _, _, _ = read_eqp_dat_file(f)
        assert bands_dft[0, 0] == pytest.approx(1.0)
        assert bands_dft[1, 0] == pytest.approx(2.0)
        assert bands_qp[0, 0]  == pytest.approx(1.1)
        assert bands_qp[1, 0]  == pytest.approx(2.2)

    def test_multi_kpoint_energy_layout(self, tmp_path):
        # Verify that energies for each kpoint are read independently
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        dft  = np.array([[1.0, 3.0], [2.0, 4.0]])   # (2 bands, 2 kpoints)
        qp   = dft + 0.5
        f = _write_eqp_dat(tmp_path, kpts, dft, qp)
        bands_dft, bands_qp, _, _, _ = read_eqp_dat_file(f)
        assert bands_dft[0, 1] == pytest.approx(3.0)   # band0, kpoint1
        assert bands_qp[1, 0]  == pytest.approx(2.5)   # band1, kpoint0
