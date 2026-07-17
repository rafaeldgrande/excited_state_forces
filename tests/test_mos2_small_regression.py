"""
Regression test: runs the current excited_forces.py against a small, real
1L-MoS2 el-ph dataset (direct fine-grid DFPT, 324 k-points, 35 bands) and
compares the RPA_diag force component against reference values produced by
old_code/excited_forces.py (the CO-paper-era implementation).

The dataset in tests/data/MoS2_small/ has real el-ph data for only one
Cartesian perturbation (S1's z-displacement); the other 8 of the 9 raw
Cartesian perturbations are exact zero. This is a deliberately adversarial
case for the acoustic sum rule (ASR), since redistributing a single nonzero
atom's response across all 3 atoms is a large correction -- it is what
exposed a real bug in old_code's impose_ASR (hardcoded k-point index 0
instead of looping over all k-points; see old_code/qe_interface_m.py, fixed
and vectorized).

tests/data/MoS2_small/reference/old_iexc{1..10}.dat holds old_code's actual
output for this dataset (10 lowest-energy excitons), regenerated after that
fix. Cross-validated against this code: max relative difference 1.16e-06
across all 10 excitons and all 9 (atom, direction) entries. To regenerate
these reference files, rerun old_code/excited_forces.py against this same
dataset (symlink elph.1.*.xml, patterns.1.xml, eigenvectors.h5, eqp.dat into
a directory with an appropriate forces.inp) rather than old_code's own slow
raw-XML path being re-exercised on every test run.
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'MoS2_small')
REFERENCE_DIR = os.path.join(DATA_DIR, 'reference')
EXCITED_FORCES_PY = os.path.join(REPO_ROOT, 'main', 'excited_forces.py')

NEW_ROW_RE = re.compile(
    r'^\s*(\d)\s+([xyz])\s+([\d.eE+-]+[+-][\d.eE+-]+j)\s+([\d.eE+-]+[+-][\d.eE+-]+j)'
)
OLD_ROW_RE = re.compile(r'^\s*(\d)\s+([xyz])\s+(\S+)\s+(\S+)')


def _run_new_code(tmp_path, iexc):
    for fname in ('elph.h5', 'eigenvectors.h5', 'eqp.dat'):
        shutil.copy(os.path.join(DATA_DIR, fname), tmp_path / fname)

    (tmp_path / 'forces.inp').write_text(
        f"iexc {iexc}\n"
        "eqp_file eqp.dat\n"
        "exciton_file eigenvectors.h5\n"
        "elph_fine_h5_file elph.h5\n"
    )

    env = dict(os.environ)
    pypath = os.pathsep.join(
        os.path.abspath(os.path.join(REPO_ROOT, sub))
        for sub in ('.', 'main', 'common', 'elph')
    )
    env['PYTHONPATH'] = pypath + os.pathsep + env.get('PYTHONPATH', '')

    result = subprocess.run(
        [sys.executable, EXCITED_FORCES_PY],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"excited_forces.py failed for iexc={iexc}:\n{result.stdout}\n{result.stderr}"
    )
    return result.stdout


def _parse_new_rpa_diag(stdout):
    """(atom, dir) -> RPA_diag real part, from new code's stdout table."""
    vals = {}
    for line in stdout.splitlines():
        m = NEW_ROW_RE.match(line)
        if m:
            vals[(int(m.group(1)), m.group(2))] = complex(m.group(3)).real
    return vals


def _parse_old_rpa_diag(path):
    """(atom, dir) -> RPA_diag, from old_code's reference .dat file."""
    vals = {}
    with open(path) as f:
        for line in f:
            m = OLD_ROW_RE.match(line)
            if m:
                vals[(int(m.group(1)), m.group(2))] = float(m.group(3))
    return vals


@pytest.mark.parametrize('iexc', range(1, 11))
def test_mos2_small_rpa_diag(tmp_path, iexc):
    stdout = _run_new_code(tmp_path, iexc)
    new_vals = _parse_new_rpa_diag(stdout)
    old_vals = _parse_old_rpa_diag(
        os.path.join(REFERENCE_DIR, f'old_iexc{iexc}.dat')
    )

    assert set(new_vals.keys()) == set(old_vals.keys())

    keys = sorted(new_vals.keys())
    actual = np.array([new_vals[k] for k in keys])
    expected = np.array([old_vals[k] for k in keys])
    np.testing.assert_allclose(
        actual, expected, rtol=1e-4, atol=1e-8,
        err_msg=f"iexc={iexc}: RPA_diag mismatch vs old_code reference (atom, dir) = {keys}",
    )
