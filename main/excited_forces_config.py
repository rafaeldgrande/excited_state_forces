# IGNORE ERRORS
IGNORE_ERRORS = False


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.constants import Ry2eV, bohr2A

# Conventions
TOL_DEG = 1e-5  # tolerance to see if two energy values are degenerate

# Dumb default parameters as a dictionary
config = {
    "iexc": 1,
    "jexc": -1,  # if it keeps to be jexc, then make jexc to iexc
    "factor_head": 1,  # factor that multiplies the head of the matrix elements of bsemat.h5
    "ncbnds_sum": -1,  # how many c/v bnds to be included in forces calculation?
    "nvbnds_sum": -1,  # if == -1, then uses all used bnds in the BSE hamiltonian
    # files and paths to be opened
    "eqp_file": 'eqp1.dat',
    "exciton_file": 'eigenvectors.h5',
    "hbse_file": 'hbse.h5',

    # conditionals
    "Calculate_Kernel": False,
    "just_RPA_diag": False,    # If true doesn't calculate forces a la David

    # Makes the excited state forces to sum to zero (obey Newton's third law), by making sum of elph matrix elems to 0.
    "acoutic_sum_rule": True,
    "use_hermicity_F": True,     # Use the fact that F_cvc'v' = conj(F_c'v'cv)
    # Reduces the number of computed terms by about half

    "log_k_points": False,     # Write k points used in BSE and DFPT calculations

    # reads Acvk coeffs from files produced by my modified version of
    # summarize_eigenvectors.x
    "read_Acvk_pos": False,
    "Acvk_directory": './', # directory where the Acvk files are

    # do not renormalize elph coefficients (make <n|dHqp|m> = <n|dHdft|m> for all n and m)
    "no_renorm_elph": False,

    # write dK (derivative of kernel) matrix elements
    "write_dK_mat": False,

    # do not check kpoints between bgw and dfpt calculations
    # trust that both codes did the calculation in the same order
    # so we do not need to map one grid in another
    "trust_kpoints_order": False,

    # run in parallel flag
    "run_parallel": False,

    # number of processes to be used in parallel
    'num_processes': 1,

    # List of ELPH to be read. If list is empty, then all are read
    "dfpt_irreps_list": [],

    # use vectorized sums
    "do_vectorized_sums": True,

    # If true read exciton pairs from file exciton_pairs.dat
    # The file needs to be something like
    # 1 1
    # 1 2
    # 1 3
    # The code will calculate the exciton-phonon coefficients <iexc|dH|jexc> for all pairs
    "read_exciton_pairs_file": False,
    "exciton_pairs": [],

    'elph_fine_h5_file': 'elph_fine.h5',  # pre-interpolated fine-grid el-ph (from interpolate_elph_bgw.py)
    'use_second_derivatives_elph_coeffs': False, # if true, use the second derivatives of elph coefficients
                                                # (g2_cond and g2_val) instead of the first derivatives to calculate the forces.
                                                # The unit here in this case is ry / bohr**2

    # Save all forces and metadata to an HDF5 file
    'save_forces_h5': False,
    'forces_h5_file': 'exc_forces.h5',

    # Finite-momentum exciton-phonon matrix element <A(Q)|dV(q)|B(Q+q)>
    # When True, exciton A is loaded from eigenvectors_A_file and B from eigenvectors_B_file.
    # The phonon momentum q is determined as Q_B - Q_A from the respective eigenvectors files.
    'finite_q_phonon': False,
    'eigenvectors_A_file': 'eigenvectors_A.h5',
    'eigenvectors_B_file': 'eigenvectors_B.h5',
}

def true_or_false(text, default_value):
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False
    else:
        return default_value

def read_input(input_file):
    # Read configuration from file and update the config dictionary

    try:
        with open(input_file) as arq_in:
            print(f'Reading input file {input_file}')
            did_I_find_the_file = True
    except Exception:
        did_I_find_the_file = False
        print(f'WARNING! - Input file {input_file} not found!')
        print('Using default values for configuration')

    if did_I_find_the_file:
        arq_in = open(input_file, 'r')
        for line in arq_in:
            linha = line.split()
            if len(linha) >= 2:
                key = linha[0]
                value = linha[1:]
                # Integer keys
                if key in [
                    'iexc', 'jexc', 'ncbnds_sum', 'nvbnds_sum', 'num_processes'
                ]:
                    config[key] = int(value[0])
                # Float keys
                elif key in ['factor_head']:
                    config[key] = float(value[0])
                # String keys
                elif key in [
                    'eqp_file', 'exciton_file', 'Acvk_directory', 'hbse_file',
                    'elph_fine_h5_file', 'forces_h5_file',
                    'eigenvectors_A_file', 'eigenvectors_B_file'
                ]:
                    config[key] = value[0]
                # Boolean keys
                elif key in [
                    'Calculate_Kernel',
                    'just_RPA_diag', 'acoutic_sum_rule', 'use_hermicity_F', 'log_k_points',
                    'read_Acvk_pos', 'no_renorm_elph',
                    'write_dK_mat', 'trust_kpoints_order', 'run_parallel',
                    'do_vectorized_sums', 'read_exciton_pairs_file',
                    'use_second_derivatives_elph_coeffs',
                    'save_forces_h5', 'finite_q_phonon'
                ]:
                    config[key] = true_or_false(value[0], config.get(key, False))
                # List of integers
                elif key == 'dfpt_irreps_list':
                    config[key] = [int(v)-1 for v in value]
                else:
                    if key[0] != '#':
                        print('Parameters not recognized in the following line:')
                        print(line)
        # Special handling for jexc default
        if config['jexc'] == -1:
            config['jexc'] = config['iexc']
