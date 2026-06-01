"""
Tests for elph/interpolate_elph_bgw.py:
  _wrap_bz, _build_kpt_map
  (interpolate_elph requires HDF5 + dtmat binary files and is not unit-tested here)
"""
import pytest
import numpy as np
from interpolate_elph_bgw import _wrap_bz, _build_kpt_map


# ─────────────────────────────────────────────────────────────
# _wrap_bz
# ─────────────────────────────────────────────────────────────

class TestWrapBZ:
    def test_zero_unchanged(self):
        assert _wrap_bz(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_half_unchanged(self):
        assert _wrap_bz(np.array([0.5]))[0] == pytest.approx(0.5)

    def test_one_folds_to_zero(self):
        assert _wrap_bz(np.array([1.0]))[0] == pytest.approx(0.0)

    def test_negative_half_maps_to_half(self):
        assert _wrap_bz(np.array([-0.5]))[0] == pytest.approx(0.5)

    def test_negative_small(self):
        assert _wrap_bz(np.array([-0.1]))[0] == pytest.approx(0.9)

    def test_large_positive(self):
        assert _wrap_bz(np.array([3.7]))[0] == pytest.approx(0.7)

    def test_3d_vector(self):
        k = np.array([-0.5, 1.0, 0.25])
        result = _wrap_bz(k)
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.25)

    def test_batch_of_kpoints(self):
        kpts = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [-0.25, 0.5, 1.5]])
        result = _wrap_bz(kpts)
        assert np.allclose(result[0], [0.0, 0.0, 0.0])
        assert np.allclose(result[1], [0.0, 0.0, 0.0])
        assert result[2, 0] == pytest.approx(0.75)
        assert result[2, 1] == pytest.approx(0.5)
        assert result[2, 2] == pytest.approx(0.5)

    def test_result_always_in_0_1(self):
        vals = np.linspace(-3.0, 3.0, 61)
        result = _wrap_bz(vals)
        assert np.all(result >= 0.0)
        assert np.all(result < 1.0 + 1e-9)


# ─────────────────────────────────────────────────────────────
# _build_kpt_map
# ─────────────────────────────────────────────────────────────

class TestBuildKptMap:
    def test_identity_mapping(self):
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        result = _build_kpt_map(kpts, kpts)
        assert list(result) == [0, 1, 2]

    def test_not_found_gives_minus_one(self):
        query = np.array([[0.3, 0.0, 0.0]])
        ref   = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        result = _build_kpt_map(query, ref)
        assert result[0] == -1

    def test_bz_translation_matches(self):
        # [1, 0, 0] should match [0, 0, 0] modulo reciprocal lattice
        query = np.array([[1.0, 0.0, 0.0]])
        ref   = np.array([[0.0, 0.0, 0.0]])
        result = _build_kpt_map(query, ref)
        assert result[0] == 0

    def test_returns_best_match_by_distance(self):
        # Two close ref points; query should match the closer one
        query = np.array([[0.1, 0.0, 0.0]])
        ref   = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        result = _build_kpt_map(query, ref, tol=0.2)
        assert result[0] == 0  # [0,0,0] is closer to [0.1,0,0]

    def test_partial_matches(self):
        query = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
        ref   = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        result = _build_kpt_map(query, ref)
        assert result[0] == 0
        assert result[1] == -1

    def test_output_length_matches_query(self):
        query = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
        ref   = np.array([[0.25, 0.0, 0.0]])
        result = _build_kpt_map(query, ref)
        assert len(result) == 3

    def test_custom_tolerance(self):
        query = np.array([[0.0, 0.0, 0.0]])
        ref   = np.array([[0.001, 0.0, 0.0]])
        # Default tol=1e-5: no match
        assert _build_kpt_map(query, ref)[0] == -1
        # Loose tol=0.01: match
        assert _build_kpt_map(query, ref, tol=0.01)[0] == 0
