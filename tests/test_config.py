"""Tests for excited_forces_config.py: true_or_false and read_input."""
import pytest
from excited_forces_config import true_or_false, read_input, config


class TestTrueOrFalse:
    def test_lowercase_true(self):
        assert true_or_false('true', False) is True

    def test_uppercase_true(self):
        assert true_or_false('TRUE', False) is True

    def test_mixed_case_true(self):
        assert true_or_false('True', False) is True

    def test_lowercase_false(self):
        assert true_or_false('false', True) is False

    def test_uppercase_false(self):
        assert true_or_false('FALSE', True) is False

    def test_unknown_returns_default_true(self):
        assert true_or_false('yes', True) is True

    def test_unknown_returns_default_false(self):
        assert true_or_false('1', False) is False

    def test_empty_string_returns_default(self):
        assert true_or_false('', False) is False


class TestReadInput:
    def test_missing_file_warns_and_keeps_defaults(self, capsys, tmp_path):
        original_iexc = config['iexc']
        read_input(str(tmp_path / 'nonexistent.inp'))
        out = capsys.readouterr().out
        assert 'WARNING' in out
        assert config['iexc'] == original_iexc

    def test_reads_integer_iexc(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('iexc 5\n')
        read_input(str(f))
        assert config['iexc'] == 5

    def test_jexc_defaults_to_iexc_when_not_set(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('iexc 7\n')
        config['jexc'] = -1   # ensure sentinel is set
        read_input(str(f))
        assert config['jexc'] == 7

    def test_jexc_explicit_overrides_default(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('iexc 2\njexc 4\n')
        read_input(str(f))
        assert config['iexc'] == 2
        assert config['jexc'] == 4

    def test_reads_boolean_true(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('Calculate_Kernel true\n')
        read_input(str(f))
        assert config['Calculate_Kernel'] is True

    def test_reads_boolean_false(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('acoutic_sum_rule false\n')
        read_input(str(f))
        assert config['acoutic_sum_rule'] is False

    def test_reads_float_factor_head(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('factor_head 2.5\n')
        read_input(str(f))
        assert config['factor_head'] == pytest.approx(2.5)

    def test_reads_string_eqp_file(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('eqp_file my_eqp.dat\n')
        read_input(str(f))
        assert config['eqp_file'] == 'my_eqp.dat'

    def test_reads_dfpt_irreps_list_one_based(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('dfpt_irreps_list 1 3 5\n')
        read_input(str(f))
        assert config['dfpt_irreps_list'] == [0, 2, 4]  # converted to 0-based

    def test_comment_lines_are_ignored(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('# this is a comment\niexc 3\n')
        read_input(str(f))
        assert config['iexc'] == 3

    def test_unrecognized_key_does_not_crash(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('totally_unknown_key somevalue\n')
        read_input(str(f))  # should not raise

    def test_reads_num_processes(self, tmp_path):
        f = tmp_path / 'forces.inp'
        f.write_text('num_processes 4\n')
        read_input(str(f))
        assert config['num_processes'] == 4
