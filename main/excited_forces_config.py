# IGNORE ERRORS
IGNORE_ERRORS = False


from scipy.constants import physical_constants

# File with variables used in excited_forces.py code
# It reads input forces.inp file
# If file not found or some parameter not find in file
# then the code uses (dumb) default values


# Conventions
TOL_DEG = 1e-5  # tolerance to see if two energy values are degenerate  ----------

# Conversion factors
Ry2eV = physical_constants["Rydberg constant times hc in eV"][0]
bohr2A = physical_constants["Bohr radius"][0]*1e10

# Ry2eV = 13.6056980659
# bohr2A = 0.529177249

# Dumb default parameters as a dictionary
config = {
    "iexc": 1,
    "jexc": -1,  # if it keeps to be jexc, then make jexc to iexc
    "factor_head": 1,  # factor that multiplies the head of the matrix elements of bsemat.h5 
    # you can find this value at 
    "ncbnds_sum": -1,  # how many c/v bnds to be included in forces calculation?
    "nvbnds_sum": -1,  # if == -1, then uses all used bnds in the BSE hamiltonian
    # files and paths to be opened
    "eqp_file": 'eqp1.dat',
    "exciton_file": 'eigenvectors.h5',
    "el_ph_dir": './',
    "kernel_file": 'bsemat.h5',

    # conditionals
    "calc_modes_basis": False,    # not being used yet
    "write_DKernel": False,    # not being used  yet
    "Calculate_Kernel": False,    # Dont change. We dont know how to work with kernel yet
    "just_RPA_diag": False,    # If true doesn't calculate forces a la David
    "report_RPA_data": False,    # report Fcvkc'v'k' matrix elements.

    # Show imaginary part of excited state force . It should be very close to 0 when iexc = jexc
    "show_imag_part": False,

    # Makes the excited state forces to sum to zero (obey Newton's third law), by making sum of elph matrix elems to 0. May want to set this variable to true
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
    # print('!!!!!!!!!!', no_renorm_elph)

    # make elph interpolation "a la BerkeleyGW code" using 
    # the "dtmat_non_bin_val" and "dtmat_non_bin_conds" files
    # those are output from the modified version of the absorption code
    "elph_fine_a_la_bgw": False,

    # parameters for interpolation 
    # in the absorption.inp file, are the following
    #  number_val_bands_coarse 10
    #  number_val_bands_fine 5
    #  number_cond_bands_coarse 8
    #  number_cond_bands_fine 5

    "ncbands_co": 0,
    "nvbands_co": 0,
    "nkpnts_co": 0,

    # write dK (derivative of kernel) matrix elements
    "write_dK_mat": False,

    # do not check kpoints between bgw and dfpt calculations
    # trust that both codes did the calculation in the same order
    # so we do not need to map one grid in another
    "trust_kpoints_order": False,

    # run in parallel flag
    "run_parallel": False   ,
    
    # number of processes to be used in parallel
    'num_processes': 1,

    # modify Acvk to be Acvk = delta_(cvk,cvk)
    # where cvk is the transition for which Acvk is originally maximum
    "use_Acvk_single_transition": False,

    # List of ELPH to be read. If list is empty, then all are read
    "dfpt_irreps_list": [],

    # If true limit sum of excited state forces to be done only on coefficients ik, ic, iv
    # listed in file "indexes_limited_sum_BSE.dat". The file has the following format
    # 1 1 1
    # 1 1 2
    # 1 1 3
    # 1 2 1
    # 1 2 2
    # 1 2 3
    # where the first collumn is the ik, the second is the ic, and the third is the iv
    # this is useful when one know a priori which transitions are the most relevant
    "limit_BSE_sum": False,

    # Number between 0.0 and 1.0
    # set the coefficients ic, iv and ik for which sum |A_cvk|^2 <= limit_BSE_sum_up_to_value
    # if it is equal to 0, then all coefficients are used
    # if it is different than 0, than make limit_BSE_sum = False, and ignore indexes_limited_sum_BSE.dat file
    "limit_BSE_sum_up_to_value": 1.0,

    # use vectorized sums
    # F_{mu,k,c1,v1,c2,v2} = A_{kcv1}^* A_{kc2v2} (g_{mu,k,c1,c2}*delta(v1,v2) - g_{mu,k,v1,v2}*delta(c1,c2))
    # F_{mu} = sum_{k,c1,v1,c2,v2} F_{mu,k,c1,v1,c2,v2}
    # Create the matrices A_{kcv1}^* A_{kc2v2} with shape nk, nc, nc, nv, nv
    # g_{mu,k,c1,c2}*delta(v1,v2) with shape nk, nc, nc, nv, nv
    # and g_{mu,k,v1,v2}*delta(c1,c2) with shape nk, nc, nc, nv, nv
    "do_vectorized_sums": True,

    # If true read exciton pairs from file exciton_pairs.dat
    # The file needs to be something like
    # 1 1 
    # 1 2
    # 1 3
    # ... 
    # The code will calculate the exciton_phoonon coefficients <iexc|dH|jexc> for all pairs
    "read_exciton_pairs_file": False,
    "exciton_pairs": [],
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
                    'iexc', 'jexc', 'ncbnds_sum', 'nvbnds_sum',
                    'ncbands_co', 'nvbands_co', 'nkpnts_co', 'num_processes'
                ]:
                    config[key] = int(value[0])
                # Float keys
                elif key in ['factor_head', 'limit_BSE_sum_up_to_value']:
                    config[key] = float(value[0])
                # String keys
                elif key in [
                    'eqp_file', 'exciton_file', 'el_ph_dir', 'dyn_file',
                    'kernel_file', 'Acvk_directory'
                ]:
                    config[key] = value[0]
                # Boolean keys
                elif key in [
                    'calc_modes_basis', 'Calculate_Kernel', 'write_DKernel',
                    'just_RPA_diag', 'report_RPA_data', 'show_imag_part',
                    'acoutic_sum_rule', 'use_hermicity_F', 'log_k_points',
                    'read_Acvk_pos', 'no_renorm_elph', 'elph_fine_a_la_bgw',
                    'write_dK_mat', 'trust_kpoints_order', 'run_parallel', 
                    'use_Acvk_single_transition',
                    'limit_BSE_sum', 'do_vectorized_sums', 'read_exciton_pairs_file'
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

