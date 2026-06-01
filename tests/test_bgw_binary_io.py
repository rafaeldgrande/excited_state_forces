"""
Tests for elph/bgw_binary_io.py:
  _scalar_dtype, _cli (error-path only)
  The binary readers (read_vmtxel, read_dtmat) require actual Fortran binary
  files and are not unit-tested here.
"""
import pytest
import numpy as np
from bgw_binary_io import _scalar_dtype, _cli


class TestScalarDtype:
    def test_complex_flavor_gives_complex128(self):
        dt = _scalar_dtype(True)
        assert dt == np.dtype(np.complex128)

    def test_real_flavor_gives_float64(self):
        dt = _scalar_dtype(False)
        assert dt == np.dtype(np.float64)

    def test_itemsize_complex(self):
        assert _scalar_dtype(True).itemsize == 16

    def test_itemsize_real(self):
        assert _scalar_dtype(False).itemsize == 8


class TestCLI:
    def test_no_args_returns_nonzero(self):
        assert _cli([]) != 0

    def test_unknown_kind_returns_nonzero(self):
        assert _cli(['unknown', 'somefile']) != 0

    def test_missing_file_raises(self):
        # vmtxel kind with a nonexistent file should raise (FortranFile opens the file)
        with pytest.raises(Exception):
            _cli(['vmtxel', '/nonexistent/file'])
