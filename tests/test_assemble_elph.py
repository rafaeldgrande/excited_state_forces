"""
Tests for elph/assemble_elph_h5.py:
  cart_to_crystal, wrap_to_bz, find_kpt_index,
  apply_acoustic_sum_rule, read_qpoints_control_ph,
  read_patterns_xml, parse_matdyn_modes
"""
import re
import pytest
import numpy as np
from assemble_elph_h5 import (
    cart_to_crystal,
    wrap_to_bz,
    find_kpt_index,
    apply_acoustic_sum_rule,
    read_qpoints_control_ph,
    read_patterns_xml,
    parse_matdyn_modes,
)


# ─────────────────────────────────────────────────────────────
# cart_to_crystal
# ─────────────────────────────────────────────────────────────

class TestCartToCrystal:
    def test_identity_bt_inv(self):
        kpts = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 0.0]])
        bt_inv = np.eye(3)
        result = cart_to_crystal(kpts, bt_inv)
        assert np.allclose(result, kpts)

    def test_simple_scale(self):
        # bt_inv = 2*I → crystal = 2 * cart
        kpts   = np.array([[1.0, 0.0, 0.0]])
        bt_inv = 2.0 * np.eye(3)
        result = cart_to_crystal(kpts, bt_inv)
        assert np.allclose(result, [[2.0, 0.0, 0.0]])

    def test_shape_preserved(self):
        kpts   = np.random.rand(5, 3)
        bt_inv = np.eye(3)
        result = cart_to_crystal(kpts, bt_inv)
        assert result.shape == (5, 3)

    def test_roundtrip_with_rotation(self):
        # A proper rotation: bt_inv = R, B = R^{-T} = R (since R^T = R^{-1})
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0, 0, 1]], dtype=float)
        bt_inv = R.T
        kpts_orig = np.random.rand(4, 3)
        kpts_crys = cart_to_crystal(kpts_orig, bt_inv)
        # Inverse transform: kpts_cart = kpts_crys @ np.linalg.inv(bt_inv).T
        kpts_back = kpts_crys @ np.linalg.inv(bt_inv).T
        assert np.allclose(kpts_back, kpts_orig)


# ─────────────────────────────────────────────────────────────
# wrap_to_bz
# ─────────────────────────────────────────────────────────────

class TestWrapToBZ:
    def test_zero(self):
        assert wrap_to_bz(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_half(self):
        assert wrap_to_bz(np.array([0.5]))[0] == pytest.approx(0.5)

    def test_one_folds_to_zero(self):
        assert wrap_to_bz(np.array([1.0]))[0] == pytest.approx(0.0)

    def test_negative_half_maps_to_half(self):
        assert wrap_to_bz(np.array([-0.5]))[0] == pytest.approx(0.5)

    def test_negative_small_value(self):
        assert wrap_to_bz(np.array([-0.1]))[0] == pytest.approx(0.9)

    def test_large_positive(self):
        assert wrap_to_bz(np.array([3.7]))[0] == pytest.approx(0.7)

    def test_3d_vector(self):
        k = np.array([-0.5, 1.0, 0.25])
        result = wrap_to_bz(k)
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(0.25)

    def test_result_always_in_0_1(self):
        vals = np.linspace(-3.0, 3.0, 61)
        result = wrap_to_bz(vals)
        assert np.all(result >= 0.0)
        assert np.all(result < 1.0 + 1e-9)


# ─────────────────────────────────────────────────────────────
# find_kpt_index
# ─────────────────────────────────────────────────────────────

class TestFindKptIndex:
    def test_exact_match(self):
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        assert find_kpt_index([0.5, 0.0, 0.0], kpts) == 1

    def test_not_found(self):
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        assert find_kpt_index([0.3, 0.0, 0.0], kpts) == -1

    def test_finds_first_occurrence(self):
        # find_kpt_index breaks early → returns first (index 0), not last
        kpts = np.array([[0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.25, 0.0, 0.0]])
        assert find_kpt_index([0.25, 0.0, 0.0], kpts) == 0

    def test_bz_equivalence(self):
        kpts = np.array([[0.0, 0.0, 0.0]])
        # k = [1, 0, 0] is equivalent mod reciprocal lattice
        assert find_kpt_index([1.0, 0.0, 0.0], kpts) == 0

    def test_gamma_point(self):
        kpts = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]])
        assert find_kpt_index([0.0, 0.0, 0.0], kpts) == 0

    def test_empty_list(self):
        kpts = np.zeros((0, 3))
        assert find_kpt_index([0.0, 0.0, 0.0], kpts) == -1


# ─────────────────────────────────────────────────────────────
# apply_acoustic_sum_rule
# ─────────────────────────────────────────────────────────────

class TestApplyAcousticSumRule:
    def test_already_zero_residual(self):
        # g already satisfies ASR → before residual ≈ 0, after ≈ 0
        nat, Nq, Nk, Nb = 2, 1, 2, 3
        g_cart = np.zeros((Nq, 3 * nat, Nk, Nb, Nb), dtype=complex)
        before, after = apply_acoustic_sum_rule(g_cart, nat)
        assert before == pytest.approx(0.0)
        assert after  == pytest.approx(0.0, abs=1e-14)

    def test_after_asr_sum_over_atoms_is_zero(self):
        nat, Nq, Nk, Nb = 2, 1, 1, 2
        g_cart = np.ones((Nq, 3 * nat, Nk, Nb, Nb), dtype=complex)
        apply_acoustic_sum_rule(g_cart, nat)
        # After ASR: sum over atoms for each direction d should be 0
        view = g_cart.reshape(Nq, nat, 3, Nk, Nb, Nb)
        atom_sum = view.sum(axis=1)   # (Nq, 3, Nk, Nb, Nb)
        assert np.allclose(np.abs(atom_sum), 0.0, atol=1e-14)

    def test_modification_is_in_place(self):
        nat = 2
        g_cart = np.ones((1, 6, 1, 2, 2), dtype=complex)
        g_orig = g_cart.copy()
        apply_acoustic_sum_rule(g_cart, nat)
        assert not np.allclose(g_cart, g_orig)  # g was modified

    def test_returns_before_and_after(self):
        nat = 3
        Nq, Nk, Nb = 1, 1, 2
        g_cart = np.ones((Nq, 3 * nat, Nk, Nb, Nb), dtype=complex)
        before, after = apply_acoustic_sum_rule(g_cart, nat)
        assert before > 0.0
        assert after  < before

    def test_wrong_npert_raises(self):
        nat = 2
        g_cart = np.ones((1, 3 * nat + 1, 1, 2, 2), dtype=complex)  # wrong Npert
        with pytest.raises(ValueError):
            apply_acoustic_sum_rule(g_cart, nat)

    def test_three_atoms_simple_case(self):
        # nat=3, one q, one k, one band: g[0, d, 0, 0, 0] = 1 for all 3 atoms in x-dir
        nat, Nq, Nk, Nb = 3, 1, 1, 1
        g_cart = np.zeros((Nq, 3 * nat, Nk, Nb, Nb), dtype=complex)
        g_cart[0, 0, 0, 0, 0] = 1.0   # atom 0, x
        g_cart[0, 3, 0, 0, 0] = 1.0   # atom 1, x
        g_cart[0, 6, 0, 0, 0] = 1.0   # atom 2, x
        apply_acoustic_sum_rule(g_cart, nat)
        # After ASR each atom's x component should be 1 - 3/3 = 0
        assert g_cart[0, 0, 0, 0, 0] == pytest.approx(0.0)
        assert g_cart[0, 3, 0, 0, 0] == pytest.approx(0.0)
        assert g_cart[0, 6, 0, 0, 0] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────
# read_qpoints_control_ph
# ─────────────────────────────────────────────────────────────

def _write_control_ph_xml(tmp_path, qpts, units='2 pi / a'):
    nums = ' '.join(f'{x:.6f}' for q in qpts for x in q)
    xml = f"""<?xml version="1.0"?>
<EXECUTION_SUMMARY>
  <Q_POINTS>
    <NUMBER_OF_Q_POINTS>{len(qpts)}</NUMBER_OF_Q_POINTS>
    <UNITS_FOR_Q-POINT UNITS="{units}" />
    <Q-POINT_COORDINATES>{nums}</Q-POINT_COORDINATES>
  </Q_POINTS>
</EXECUTION_SUMMARY>"""
    path = tmp_path / 'control_ph.xml'
    path.write_text(xml)
    return str(path)


class TestReadQpointsControlPH:
    def test_single_gamma(self, tmp_path):
        path = _write_control_ph_xml(tmp_path, [[0.0, 0.0, 0.0]])
        qpts = read_qpoints_control_ph(path)
        assert qpts.shape == (1, 3)
        assert np.allclose(qpts[0], [0.0, 0.0, 0.0])

    def test_two_qpoints(self, tmp_path):
        pts = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
        path = _write_control_ph_xml(tmp_path, pts)
        qpts = read_qpoints_control_ph(path)
        assert qpts.shape == (2, 3)
        assert np.allclose(qpts[0], pts[0])
        assert np.allclose(qpts[1], pts[1])

    def test_dtype_is_float64(self, tmp_path):
        path = _write_control_ph_xml(tmp_path, [[0.0, 0.0, 0.0]])
        qpts = read_qpoints_control_ph(path)
        assert qpts.dtype == np.float64

    def test_missing_q_points_block_raises(self, tmp_path):
        xml = '<EXECUTION_SUMMARY><SOMETHING/></EXECUTION_SUMMARY>'
        path = tmp_path / 'bad.xml'
        path.write_text(xml)
        with pytest.raises(ValueError):
            read_qpoints_control_ph(str(path))


# ─────────────────────────────────────────────────────────────
# read_patterns_xml
# ─────────────────────────────────────────────────────────────

def _write_cartesian_patterns_xml(tmp_path, nat):
    """Write a patterns.xml where each perturbation is a pure Cartesian unit vector.
    IRREPS_INFO must be a *child* of the root element (not the root itself).
    """
    npert = 3 * nat
    lines = [
        '<?xml version="1.0"?>',
        '<PATTERNS_FILE>',       # wrapper root element
        '  <IRREPS_INFO>',
        f'    <NUMBER_IRR_REP>1</NUMBER_IRR_REP>',
        '    <REPRESENTION.1>',
        f'      <NUMBER_OF_PERTURBATIONS>{npert}</NUMBER_OF_PERTURBATIONS>',
    ]
    for ipert in range(npert):
        vec = np.zeros(npert)
        vec[ipert] = 1.0
        nums = ' '.join(f'{x:.1f} 0.0' for x in vec)
        lines += [
            f'      <PERTURBATION.{ipert + 1}>',
            f'        <DISPLACEMENT_PATTERN>{nums}</DISPLACEMENT_PATTERN>',
            f'      </PERTURBATION.{ipert + 1}>',
        ]
    lines += ['    </REPRESENTION.1>', '  </IRREPS_INFO>', '</PATTERNS_FILE>']
    path = tmp_path / 'patterns.1.xml'
    path.write_text('\n'.join(lines))
    return str(path)


class TestReadPatternsXML:
    def test_cartesian_patterns_give_identity(self, tmp_path):
        nat = 2
        path = _write_cartesian_patterns_xml(tmp_path, nat)
        U = read_patterns_xml(path, nat)
        assert U.shape == (6, 6)
        assert np.allclose(U, np.eye(6))

    def test_result_is_unitary(self, tmp_path):
        nat = 1   # 3 perturbations
        path = _write_cartesian_patterns_xml(tmp_path, nat)
        U = read_patterns_xml(path, nat)
        err = np.max(np.abs(U @ U.conj().T - np.eye(3)))
        assert err < 1e-10

    def test_shape_is_npert_by_npert(self, tmp_path):
        nat = 2
        path = _write_cartesian_patterns_xml(tmp_path, nat)
        U = read_patterns_xml(path, nat)
        assert U.shape == (3 * nat, 3 * nat)


# ─────────────────────────────────────────────────────────────
# parse_matdyn_modes
# ─────────────────────────────────────────────────────────────

def _write_matdyn_modes(tmp_path, nat, nq=1):
    """Write a minimal matdyn.modes file with identity-like eigenvectors."""
    nmodes = 3 * nat
    lines = []
    for iq in range(nq):
        lines.append(f' q = {iq * 0.5:.6f} 0.000000 0.000000')
        for nu in range(nmodes):
            freq_thz = float(nu)
            freq_cm1 = freq_thz * 33.356  # approx cm^-1
            lines.append(f' freq (  {nu + 1}) =  {freq_thz:.4f} [THz] =  {freq_cm1:.4f} [cm-1]')
            # eigenvector for nat atoms; only one component nonzero
            for iat in range(nat):
                if iat * 3 + (nu % 3) == nu % nmodes:
                    vals = [0.0] * 3
                    vals[nu % 3] = 1.0
                else:
                    vals = [0.0, 0.0, 0.0]
                parts = ' '.join(f'( {v:.6f} 0.000000' for v in vals) + ' )'
                # Format each atom as: ( Re Im  Re Im  Re Im )
                parts = '( ' + '  '.join(
                    f'{v:.6f}  0.000000' for v in vals) + ' )'
                lines.append(f'   {parts}')
    path = tmp_path / 'matdyn.modes'
    path.write_text('\n'.join(lines) + '\n')
    return str(path)


class TestParseMatdynModes:
    def test_output_shapes_single_q(self, tmp_path):
        nat = 1
        path = _write_matdyn_modes(tmp_path, nat, nq=1)
        qpts, freqs, evecs = parse_matdyn_modes(path, nat)
        nmodes = 3 * nat
        assert qpts.shape  == (1, 3)
        assert freqs.shape == (1, nmodes)
        assert evecs.shape == (1, nmodes, nat, 3)

    def test_output_shapes_multiple_q(self, tmp_path):
        nat = 2
        nq  = 3
        path = _write_matdyn_modes(tmp_path, nat, nq=nq)
        qpts, freqs, evecs = parse_matdyn_modes(path, nat)
        nmodes = 3 * nat
        assert qpts.shape  == (nq, 3)
        assert freqs.shape == (nq, nmodes)
        assert evecs.shape == (nq, nmodes, nat, 3)

    def test_eigenvectors_are_normalized(self, tmp_path):
        nat = 1
        path = _write_matdyn_modes(tmp_path, nat, nq=1)
        _, _, evecs = parse_matdyn_modes(path, nat)
        nmodes = 3 * nat
        for nu in range(nmodes):
            norm = np.sqrt(np.sum(np.abs(evecs[0, nu]) ** 2))
            assert norm == pytest.approx(1.0, abs=1e-10)

    def test_q_coordinates_parsed(self, tmp_path):
        nat = 1
        path = _write_matdyn_modes(tmp_path, nat, nq=2)
        qpts, _, _ = parse_matdyn_modes(path, nat)
        assert qpts[0, 0] == pytest.approx(0.0)
        assert qpts[1, 0] == pytest.approx(0.5)

    def test_frequencies_nonnegative_first_mode(self, tmp_path):
        nat = 1
        path = _write_matdyn_modes(tmp_path, nat, nq=1)
        _, freqs, _ = parse_matdyn_modes(path, nat)
        # First mode has freq_thz=0 → freq_cm1=0
        assert freqs[0, 0] == pytest.approx(0.0, abs=1e-3)

    def test_wrong_mode_count_raises(self, tmp_path):
        nat = 2
        # Write only 1 mode instead of 6
        lines = [' q = 0.000000 0.000000 0.000000',
                 ' freq (  1) =  1.0000 [THz] =  33.3560 [cm-1]']
        for iat in range(nat):
            lines.append('   ( 1.000000  0.000000  0.000000  0.000000  0.000000  0.000000 )')
        path = tmp_path / 'bad_modes.modes'
        path.write_text('\n'.join(lines) + '\n')
        with pytest.raises(ValueError):
            parse_matdyn_modes(str(path), nat)
