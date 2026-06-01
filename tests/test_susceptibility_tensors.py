"""
Tests for resonant_raman/ tensor calculation functions.

All resonant_raman scripts run argparse + file-loading at module level, so they
cannot be imported cleanly.  We use _partial_import(): set sys.argv to the script
name so argparse uses defaults, then catch the FileNotFoundError that occurs when
the script tries to open missing data files.  The functions defined *before* that
error are still present on the module object and can be called after injecting the
necessary globals.
"""
import sys
import importlib.util
from pathlib import Path
import pytest
import numpy as np
import h5py

RR_DIR = Path(__file__).parent.parent / 'resonant_raman'


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _partial_import(filepath, module_name):
    """Load a script up to the first missing-file error; return the module object."""
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    mod  = importlib.util.module_from_spec(spec)
    old_argv = sys.argv[:]
    sys.argv  = [str(filepath)]
    try:
        spec.loader.exec_module(mod)
    except (FileNotFoundError, OSError, SystemExit, KeyError):
        pass
    finally:
        sys.argv = old_argv
    return mod


def _make_bse1st_globals(mod, Nexc=5, Nmodes=3, Nfreq=6, seed=42):
    """Inject BSE first-order globals into a partially loaded first_order module."""
    rng = np.random.default_rng(seed)
    mod.Nexc     = Nexc
    mod.Nmodes   = Nmodes
    mod.Nfreq    = Nfreq
    mod.Ex       = np.linspace(1.0, 3.0, Nfreq)
    mod.gamma    = 0.05
    mod.freqs_eV = rng.random(Nmodes) * 0.2 + 0.05
    mod.exc_energies = rng.random(Nexc) * 2.0 + 1.0
    raw = rng.random((Nmodes, Nexc, Nexc)) + 1j * rng.random((Nmodes, Nexc, Nexc))
    mod.exc_ph = (raw + raw.conj().transpose(0, 2, 1)) / 2    # Hermitian
    pa = rng.random(Nexc) + 1j * rng.random(Nexc)
    pb = rng.random(Nexc) + 1j * rng.random(Nexc)
    pc = rng.random(Nexc) + 1j * rng.random(Nexc)
    mod.pos_operator_list = [pa, pb, pc]


def _make_ipa_first_order_globals(mod, nk=2, nc=3, nv=2, Nmodes=3, Nfreq=5, seed=0):
    """Inject IPA first-order globals into a partially loaded IPA module."""
    rng = np.random.default_rng(seed)
    mod.nk_elph  = nk
    mod.nc_elph  = nc
    mod.nv_elph  = nv
    mod.Nmodes   = Nmodes
    mod.Nfreq    = Nfreq
    mod.Ex       = np.linspace(1.0, 3.0, Nfreq)
    mod.gamma    = 0.05
    mod.freqs_eV = rng.random(Nmodes) * 0.2 + 0.05
    mod.DeltaE   = rng.random((nk, nc, nv)) * 2.0 + 1.0   # (nk, nc, nv)
    mod.g_cond   = rng.random((Nmodes, nk, nc, nc)) + 1j * rng.random((Nmodes, nk, nc, nc))
    mod.g_val    = rng.random((Nmodes, nk, nv, nv)) + 1j * rng.random((Nmodes, nk, nv, nv))
    mod.g_cond_dag = mod.g_cond.conj().transpose(0, 1, 3, 2)
    mod.g_val_dag  = mod.g_val.conj().transpose(0, 1, 3, 2)
    pa = rng.random((nk, nc, nv)) + 1j * rng.random((nk, nc, nv))
    pb = rng.random((nk, nc, nv)) + 1j * rng.random((nk, nc, nv))
    pc = rng.random((nk, nc, nv)) + 1j * rng.random((nk, nc, nv))
    mod.pos_operator_list = [pa, pb, pc]


def _make_ipa_second_order_globals(mod, nk=2, nc=2, nv=2, Nmodes=2, Nfreq=4, seed=7):
    """Inject IPA second-order globals (superset of first-order)."""
    _make_ipa_first_order_globals(mod, nk, nc, nv, Nmodes, Nfreq, seed)


# ─────────────────────────────────────────────────────────────
# delta_E  (pure function, available from IPA script before file loading)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def ipa_module():
    return _partial_import(RR_DIR / 'susceptibility_tensors_IPA.py', '_ipa_test')


class TestDeltaE:
    def test_shape(self, ipa_module):
        delta_E = ipa_module.delta_E
        Econd = np.ones((3, 4))   # (nc, nk)
        Eval  = np.ones((2, 4))   # (nv, nk)
        result = delta_E(Econd, Eval)
        assert result.shape == (4, 3, 2)   # (nk, nc, nv)

    def test_values(self, ipa_module):
        delta_E = ipa_module.delta_E
        Econd = np.array([[2.0, 4.0]])   # nc=1, nk=2
        Eval  = np.array([[1.0, 2.0]])   # nv=1, nk=2
        # output[ik, ic, iv] = Econd[ic, ik] - Eval[iv, ik]
        result = delta_E(Econd, Eval)
        assert result[0, 0, 0] == pytest.approx(2.0 - 1.0)   # ik=0
        assert result[1, 0, 0] == pytest.approx(4.0 - 2.0)   # ik=1

    def test_multiple_bands(self, ipa_module):
        delta_E = ipa_module.delta_E
        Econd = np.array([[3.0], [4.0]])   # nc=2, nk=1
        Eval  = np.array([[1.0], [2.0]])   # nv=2, nk=1
        result = delta_E(Econd, Eval)
        assert result.shape == (1, 2, 2)
        assert result[0, 0, 0] == pytest.approx(3.0 - 1.0)
        assert result[0, 0, 1] == pytest.approx(3.0 - 2.0)
        assert result[0, 1, 0] == pytest.approx(4.0 - 1.0)
        assert result[0, 1, 1] == pytest.approx(4.0 - 2.0)

    def test_antisymmetric_in_band_swap(self, ipa_module):
        delta_E = ipa_module.delta_E
        Econd = np.random.rand(3, 5)
        Eval  = np.random.rand(2, 5)
        r  = delta_E(Econd, Eval)   # (nk, nc, nv)
        r2 = delta_E(Eval,  Econd)  # swapped: (nk, nv, nc)
        # r[ik, ic, iv] + r2[ik, iv, ic] should be 0
        assert np.allclose(r + r2.transpose(0, 2, 1), 0.0)


# ─────────────────────────────────────────────────────────────
# BSE first-order tensor consistency
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def bse1st_module():
    mod = _partial_import(RR_DIR / 'susceptibility_tensors_first_order.py', '_bse1st_test')
    assert hasattr(mod, 'calculate_tensor_not_vectorized'), \
        "Partial import did not capture calculate_tensor_not_vectorized"
    _make_bse1st_globals(mod)
    return mod


class TestBSEFirstOrderConsistency:
    """All three implementation flavors must give identical (d2, d3) arrays."""

    def test_vectorize_excitons_matches_not_vectorized(self, bse1st_module):
        mod = bse1st_module
        d2_0, d3_0 = mod.calculate_tensor_not_vectorized(0, 1)
        d2_1, d3_1 = mod.calculate_tensor_vectorize_over_excitons(0, 1)
        assert np.allclose(d2_0, d2_1, atol=1e-10), \
            f"d2 mismatch: max|diff|={np.max(np.abs(d2_0 - d2_1)):.2e}"
        assert np.allclose(d3_0, d3_1, atol=1e-10), \
            f"d3 mismatch: max|diff|={np.max(np.abs(d3_0 - d3_1)):.2e}"

    def test_fully_vectorized_matches_not_vectorized(self, bse1st_module):
        mod = bse1st_module
        d2_0, d3_0 = mod.calculate_tensor_not_vectorized(0, 2)
        d2_2, d3_2 = mod.calculate_tensor_vectorized_over_modes_and_excitons(0, 2)
        assert np.allclose(d2_0, d2_2, atol=1e-10), \
            f"d2 mismatch: max|diff|={np.max(np.abs(d2_0 - d2_2)):.2e}"
        assert np.allclose(d3_0, d3_2, atol=1e-10), \
            f"d3 mismatch: max|diff|={np.max(np.abs(d3_0 - d3_2)):.2e}"

    def test_d2_is_diagonal_subset_of_d3(self, bse1st_module):
        # d2 uses only diagonal exc_ph[m,s,s]; d3 uses the full matrix.
        # If exc_ph is diagonal, d2 == d3.
        mod = bse1st_module
        # Temporarily make exc_ph diagonal
        exc_ph_orig = mod.exc_ph.copy()
        for im in range(mod.Nmodes):
            diag = np.diag(mod.exc_ph[im]).copy()
            mod.exc_ph[im] = np.diag(diag)
        d2, d3 = mod.calculate_tensor_not_vectorized(0, 0)
        assert np.allclose(d2, d3, atol=1e-10)
        mod.exc_ph = exc_ph_orig   # restore

    def test_output_shape(self, bse1st_module):
        mod = bse1st_module
        d2, d3 = mod.calculate_tensor_not_vectorized(0, 0)
        assert d2.shape == (mod.Nmodes, mod.Nfreq)
        assert d3.shape == (mod.Nmodes, mod.Nfreq)


# ─────────────────────────────────────────────────────────────
# _load_exc_ph_h5  (from susceptibility_tensors_second_order.py)
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def bse2nd_module():
    mod = _partial_import(RR_DIR / 'susceptibility_tensors_second_order.py', '_bse2nd_test')
    assert hasattr(mod, '_load_exc_ph_h5'), \
        "Partial import did not capture _load_exc_ph_h5"
    return mod


def _write_exc_ph_h5(path, pairs, forces, ph_freqs=None):
    """Write a minimal exc_forces h5 file in the new format. path may be a Path or str."""
    with h5py.File(str(path), 'w') as hf:
        hf.create_dataset('exciton_pairs', data=np.array(pairs, dtype=np.int32))
        hf.create_dataset('forces/ph/RPA',  data=np.array(forces, dtype=float))
        if ph_freqs is not None:
            grp = hf.require_group('system')
            grp.create_dataset('phonon_frequencies', data=ph_freqs)
    return str(path)


class TestLoadExcPhH5:
    def test_diagonal_pair(self, tmp_path, bse2nd_module):
        _load = bse2nd_module._load_exc_ph_h5
        Nm = 3
        pairs  = [[1, 1], [2, 2]]
        forces = np.ones((2, Nm))
        path = _write_exc_ph_h5(tmp_path / 'diag_exc_ph.h5', pairs, forces)
        mat, _, _, _ = _load(path, hermitian=True)
        assert mat.shape == (Nm, 2, 2)
        # F = -<A|dH|B> → mat = -forces
        assert np.allclose(mat[:, 0, 0], -1.0)
        assert np.allclose(mat[:, 1, 1], -1.0)

    def test_hermitian_symmetry_filled(self, tmp_path, bse2nd_module):
        _load = bse2nd_module._load_exc_ph_h5
        Nm = 2
        pairs  = [[1, 1], [1, 2]]
        forces_real = np.ones((2, Nm))
        path = _write_exc_ph_h5(tmp_path / 'herm_exc_ph.h5', pairs, forces_real)
        mat, _, _, _ = _load(str(path), hermitian=True)
        # mat[:, 0, 1] should equal mat[:, 1, 0].conj() (since pair (1,2) fills both)
        assert np.allclose(mat[:, 0, 1], mat[:, 1, 0].conj())

    def test_off_diagonal_zero_without_pair(self, tmp_path, bse2nd_module):
        _load = bse2nd_module._load_exc_ph_h5
        Nm = 2
        # Give pairs (1,1) and (2,2) only — no off-diagonal pair (1,2)
        pairs  = [[1, 1], [2, 2]]
        forces = np.ones((2, Nm))
        path = _write_exc_ph_h5(tmp_path / 'offdiag_exc_ph.h5', pairs, forces)
        mat, _, _, _ = _load(str(path), hermitian=True)
        assert mat.shape == (Nm, 2, 2)
        # Off-diagonal slots not provided → should remain zero
        assert np.allclose(mat[:, 0, 1], 0.0)
        assert np.allclose(mat[:, 1, 0], 0.0)

    def test_phonon_frequencies_loaded(self, tmp_path, bse2nd_module):
        _load = bse2nd_module._load_exc_ph_h5
        Nm = 4
        freqs = np.array([100.0, 200.0, 300.0, 400.0])
        pairs  = [[1, 1]]
        forces = np.ones((1, Nm))
        path = _write_exc_ph_h5(tmp_path / 'freqs_exc_ph.h5', pairs, forces, freqs)
        _, ph_freqs, _, _ = _load(str(path))
        assert np.allclose(ph_freqs, freqs)


# ─────────────────────────────────────────────────────────────
# IPA first-order tensor consistency
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def ipa1st_module(ipa_module):
    _make_ipa_first_order_globals(ipa_module)
    return ipa_module


class TestIPAFirstOrderConsistency:
    """All three IPA first-order implementations must agree."""

    def test_vectorized_over_kcv_matches_not_vectorized(self, ipa1st_module):
        mod = ipa1st_module
        r0 = mod.calculate_tensor_first_order_not_vectorized(0, 1)
        r1 = mod.calculate_tensor_first_order_vectorized_over_kcv(0, 1)
        assert np.allclose(r0, r1, atol=1e-10), \
            f"max|diff|={np.max(np.abs(r0 - r1)):.2e}"

    def test_fully_vectorized_matches_not_vectorized(self, ipa1st_module):
        mod = ipa1st_module
        r0 = mod.calculate_tensor_first_order_not_vectorized(0, 1)
        r2 = mod.calculate_tensor_first_order_vectorized_over_modes_and_kcv(0, 1)
        assert np.allclose(r0, r2, atol=1e-10), \
            f"max|diff|={np.max(np.abs(r0 - r2)):.2e}"

    def test_output_shape(self, ipa1st_module):
        mod = ipa1st_module
        r = mod.calculate_tensor_first_order_not_vectorized(0, 0)
        assert r.shape == (mod.Nmodes, mod.Nfreq)

    def test_zero_result_for_zero_pos_operator(self, ipa1st_module):
        mod = ipa1st_module
        orig = mod.pos_operator_list[0].copy()
        mod.pos_operator_list[0] = np.zeros_like(orig)
        r = mod.calculate_tensor_first_order_not_vectorized(0, 1)
        assert np.allclose(r, 0.0)
        mod.pos_operator_list[0] = orig


# ─────────────────────────────────────────────────────────────
# IPA second-order tensor consistency
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def ipa2nd_module(ipa_module):
    _make_ipa_second_order_globals(ipa_module)
    return ipa_module


class TestIPASecondOrderConsistency:
    """The three IPA second-order implementations must agree."""

    def test_vectorized_over_kcv_matches_not_vectorized(self, ipa2nd_module):
        mod = ipa2nd_module
        r0 = mod.calculate_tensor_second_order_not_vectorized(0, 1)
        r1 = mod.calculate_tensor_second_order_vectorized_over_kcv(0, 1)
        assert np.allclose(r0, r1, atol=1e-10), \
            f"max|diff|={np.max(np.abs(r0 - r1)):.2e}"

    def test_fully_vectorized_matches_not_vectorized(self, ipa2nd_module):
        mod = ipa2nd_module
        r0 = mod.calculate_tensor_second_order_not_vectorized(0, 1)
        r2 = mod.calculate_tensor_second_order_vectorized_over_jmode_and_kcv(0, 1)
        assert np.allclose(r0, r2, atol=1e-10), \
            f"max|diff|={np.max(np.abs(r0 - r2)):.2e}"

    def test_output_shape(self, ipa2nd_module):
        mod = ipa2nd_module
        r = mod.calculate_tensor_second_order_not_vectorized(0, 0)
        assert r.shape == (mod.Nmodes, mod.Nmodes, mod.Nfreq)
