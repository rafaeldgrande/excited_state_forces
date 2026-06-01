"""Tests for excited_forces_classes.py: Parameters_BSE, Parameters_MF, Parameters_ELPH."""
import pytest
import numpy as np
from excited_forces_classes import Parameters_BSE, Parameters_MF, Parameters_ELPH


class TestParametersBSE:
    def setup_method(self):
        self.kpts = np.zeros((4, 3))
        self.rec = np.eye(3)

    def test_attributes_stored(self):
        p = Parameters_BSE(4, self.kpts, 3, 2, 5, 3, 2, self.rec)
        assert p.Nkpoints_BSE == 4
        assert p.Ncbnds == 3
        assert p.Nvbnds == 2
        assert p.Nval == 5
        assert p.Ncbnds_sum == 3
        assert p.Nvbnds_sum == 2

    def test_kpoints_reference(self):
        p = Parameters_BSE(4, self.kpts, 3, 2, 5, 3, 2, self.rec)
        assert np.allclose(p.Kpoints_BSE, self.kpts)

    def test_rec_cell_vecs_reference(self):
        p = Parameters_BSE(4, self.kpts, 3, 2, 5, 3, 2, self.rec)
        assert np.allclose(p.rec_cell_vecs, np.eye(3))


class TestParametersMF:
    def test_nmodes_is_3nat(self):
        pos = np.zeros((6, 3))
        vecs = np.eye(3)
        p = Parameters_MF(6, pos, vecs, 200.0, 7.5)
        assert p.Nmodes == 18

    def test_attributes_stored(self):
        pos = np.zeros((2, 3))
        vecs = np.eye(3) * 2.0
        p = Parameters_MF(2, pos, vecs, 50.0, 3.0)
        assert p.Nat == 2
        assert p.cell_vol == pytest.approx(50.0)
        assert p.alat == pytest.approx(3.0)
        assert np.allclose(p.cell_vecs, np.eye(3) * 2.0)


class TestParametersELPH:
    def test_attributes_stored(self):
        kpts = np.zeros((8, 3))
        p = Parameters_ELPH(8, kpts)
        assert p.Nkpoints_DFPT == 8
        assert np.allclose(p.Kpoints_DFPT, kpts)

    def test_note_typo_in_param_name(self):
        # Constructor argument is 'Nkpoints_DPFT' (typo) but attribute is 'Nkpoints_DFPT'
        kpts = np.zeros((3, 3))
        p = Parameters_ELPH(3, kpts)
        assert hasattr(p, 'Nkpoints_DFPT')
