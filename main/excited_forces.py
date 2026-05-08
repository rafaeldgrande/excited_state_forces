
verbosity = 'high'

# excited state forces modules
from modules_to_import import *
from excited_forces_config import *
from bgw_interface_m import *
from qe_interface_m import *
from excited_forces_m import *
from excited_forces_classes import *

# trace ram
tracemalloc.start()

# functions

def translate_bse_to_dfpt_k_points():

    ikBSE_to_ikDFPT = []
    # This list shows which k point from BSE corresponds to which
    # point from DFPT calculation.
    # ikBSE_to_ikDFPT[ikBSE] = ikDFPT
    # Means that the k point from eigenvectors.h5 with index ikBSE corresponds to
    # the k point with index ikDFPT from DFPT calculation
    # We also can trust that everything is ok, and they have 1 to 1 correspondence
    # and are given in the same order
    # This is done with the flag trust_kpoints_order = True
    
    if trust_kpoints_order == False:
        #debug
        arq_teste = open('Kpoints_in_elph_eigvecs_cart_basis', 'w')

        for ik in range(Nkpoints_BSE):

            # getting vectors from eigenvectors.h5 file in latt vectors basis
            a1, a2, a3 = Kpoints_BSE[ik]

            # putting the vector in the first Brillouin zone
            a1 = correct_comp_vector(a1)
            a2 = correct_comp_vector(a2)
            a3 = correct_comp_vector(a3)

            # vector in cartesian basis
            # vec_eigvecs = a1 * rec_cell_vecs[0] + a2 * rec_cell_vecs[1] + a3 * rec_cell_vecs[2]
            vec_eigvecs = np.array([a1, a2, a3])
            
            arq_teste.write(f'{vec_eigvecs[0]:.9f}   {vec_eigvecs[1]:.9f}   {vec_eigvecs[2]:.9f}\n')
            

            found_or_not = find_kpoint(vec_eigvecs, Kpoints_in_elph_file_frac)
            # if found the vec_eigvecs in the Kpoints_in_elph_file_frac, then returns
            # the index in the Kpoints_in_elph_file_frac.
            # if did not find it, then returns -1

            # the conversion list from one to another
            ikBSE_to_ikDFPT.append(found_or_not)

        # debug
        arq_teste.close()
        
    else:
        for ik in range(len(Kpoints_in_elph_file_frac)):
            ikBSE_to_ikDFPT.append(ik)
    
    return ikBSE_to_ikDFPT


def check_k_points_BSE_DFPT():
    # Checking if any kpoint is missing
    flag_missing_kpoints = False
    flag_repeated_kpoints = False

    if ikBSE_to_ikDFPT.count(-1) > 0:

        flag_missing_kpoints = True

        print('WARNING! Some k points from eigenvecs file were not found in the grid used in the DFPT calculation!')
        print(f'Total number of missing k points {ikBSE_to_ikDFPT.count(-1)}')
        print('The missing k points in DFPT are (in reciprocal lattice basis):')
        for ik in range(Nkpoints_BSE):
            if ikBSE_to_ikDFPT[ik] == -1:
                print(Kpoints_BSE[ik])

    # Checking if any kpoint is reported more than once

    for ikBSE in range(Nkpoints_BSE):
        how_many_times = 0
        for ikBSE2 in range(Nkpoints_BSE):
            if np.linalg.norm(Kpoints_BSE[ikBSE] - Kpoints_BSE[ikBSE2]) <= TOL_DEG:
                how_many_times += 1
        if how_many_times > 1:
            print(
                f'WARNING!    This k point appear more than once: {Kpoints_BSE[ikBSE]} ')
            flag_repeated_kpoints = True

    if flag_missing_kpoints == False and flag_repeated_kpoints == False:
        print('Found no problem for k points from both DFPT and BSE calculations')
    else:
        if IGNORE_ERRORS == False:
            print('Quiting program! Please check the above warnings!')
            quit()
        else:
            print('Continuing calculation regardless of that!')


def load_exciton_coeffs(iexc, jexc, verbose=False):
    time0 = time.clock_gettime(0)
    Akcv, Bkcv = get_exciton_coeffs(iexc, jexc)
    # report_expected_energies_master(iexc, jexc, Eqp_cond, Eqp_val, Akcv, OmegaA, Bkcv, OmegaB)

    # FIXME: implement this later. This limits the sums.
    # if limit_BSE_sum_up_to_value < 1.0:
    #     top_indexes_Akcv = summarize_Acvk(Akcv, BSE_params, limit_BSE_sum_up_to_value)
    #     if iexc != jexc:
    #         top_indexes_Bkcv = summarize_Acvk(Bkcv, BSE_params, limit_BSE_sum_up_to_value)

    #         # Convert lists of lists to sets of tuples
    #         set_A = set(tuple(item) for item in top_indexes_Akcv)
    #         set_B = set(tuple(item) for item in top_indexes_Bkcv)

    #         # Merge the sets to eliminate duplicates
    #         merged_set = set_A | set_B  # or set_A.union(set_B)

    #         # Convert the set back to a list of lists
    #         indexes_limited_BSE_sum = [list(item) for item in merged_set]
            
    #     else:
    #         indexes_limited_BSE_sum = top_indexes_Akcv

    # summarizing Akcv information  
    # summarize_Acvk(Akcv, BSE_params.Kpoints_BSE)

    delta_time = time.clock_gettime(0) - time0
    return Akcv, Bkcv, delta_time

def load_elph_cartesian_h5(h5_path, iq=0):
    """
    Load electron-phonon matrix elements in Cartesian displacement basis from
    the HDF5 file produced by assemble_elph_h5.py.

    Parameters
    ----------
    h5_path : str
    iq      : int  q-point index (0-based); default 0 = Gamma

    Returns
    -------
    g_cart          : (Npert, Nk, Nbnds, Nbnds) complex128   [Ry/bohr]
    kpoints_crystal : (Nk, 3) float64   DFT k-points in crystal (fractional) coords
    """
    with h5py.File(h5_path, 'r') as fh:
        g_cart          = fh['g'][iq]                    # (Npert, Nk, Nbnds, Nbnds)
        kpoints_crystal = fh['kpoints_dft_crystal'][:]   # (Nk, 3)
    return g_cart, kpoints_crystal


def f_disp_to_cart_basis(f_dis_basis, Displacements):
    '''
    Convert forces from displacement basis to cartesian basis.
    also convert from Ry/bohr to eV/ang. The minus sign comes from F=-dV/du
    input f_dis_basis is shape (Nmodes,) and is in Ry/bohr
    output f_cart is shape (Nat, 3) and is in eV/ang
    '''
    f_cart = np.zeros((Nat, 3), dtype=complex)
    
    for iatom in range(Nat):
        for imode in range(Nmodes):
            f_cart[iatom] = f_cart[iatom] + Displacements[imode, iatom] * f_dis_basis[imode]

    if use_second_derivatives_elph_coeffs == False:
        return -f_cart * Ry2eV / bohr2A
    else:
        return -f_cart * Ry2eV / bohr2A**2

def report_forces_ph(iexc, jexc, f_rpa_diag, f_rpa, f_kernel, frequencies, verbose=False):
    """Write phonon-mode basis forces: one row per mode with frequency and force values."""
    F_kernel = f_kernel if f_kernel is not None else np.zeros_like(f_rpa_diag)
    iexc_name = excitons_to_be_loaded[iexc]
    jexc_name = excitons_to_be_loaded[jexc]
    fname = f'exc_forces_{iexc_name}_{jexc_name}_ph.dat'
    unit_label = '(eV/ang^2)' if use_second_derivatives_elph_coeffs else '(eV/ang)'
    header = (
        f'# Phonon-mode basis excited-state forces {unit_label}\n'
        f'# RPA_diag: eq.(1) arxiv:2502.05144, d/dr <kcv|K^eh|k\'c\'v\'> = 0\n'
        f'# RPA:      eq.(3) arxiv:2502.05144, <kcv|d K^eh/dr|k\'c\'v\'> = 0\n'
        f'# RPA_diag_plus_Kernel: eq.(1) with kernel correction\n'
        f'# {"mode":<6} {"freq(cm-1)":<14} {"RPA_diag":<28} {"RPA":<28} {"RPA_diag_plus_Kernel"}\n'
    )
    with open(fname, 'w') as fout:
        fout.write(header)
        if verbose:
            print(header, end='')
        for inu in range(len(frequencies)):
            line = (f'{inu+1:<6} {frequencies[inu]:<14.4f} '
                    f'{f_rpa_diag[inu]:<28.8f} '
                    f'{f_rpa[inu]:<28.8f} '
                    f'{(F_kernel + f_rpa_diag)[inu]:<28.8f}')
            fout.write(line + '\n')
            if verbose:
                print(line)

def report_forces(iexc, jexc, F_RPA_diag, F_RPA, F_kernel, suffix='ph', verbose=False):
    # Convert from Ry/bohr to eV/A. Minus sign comes from F=-dV/du

    # F_RPA_diag    = F_RPA_diag 
    # F_RPA = F_RPA 
    F_kernel = F_kernel if F_kernel is not None else np.zeros_like(F_RPA_diag)
    # Sum_DKernel = -Sum_DKernel * Ry2eV / bohr2A

    # Warn if imag part is too big (>= 10^-6)

    # if max(abs(np.imag(F_RPA_diag))) >= 1e-6 and iexc == jexc:
    #     print('WARNING: Imaginary part of kinectic diagonal forces >= 10^-6 eV/angs!')

    # if max(abs(np.imag(F_RPA))) >= 1e-6 and iexc == jexc:
    #     print('WARNING: Imaginary part of kinectic offdiagonal forces >= 10^-6 eV/angs!')

    # if max(abs(np.imag(F_kernel))) >= 1e-6 and iexc == jexc:
    #     print('WARNING: Imaginary part of Kernel forces >= 10^-6 eV/angs!')

    # Reporting forces in cartesian basis
    DIRECTION = ['x', 'y', 'z']

    iexc_name = excitons_to_be_loaded[iexc]
    jexc_name = excitons_to_be_loaded[jexc]
    arq_out_name = f'exc_forces_{iexc_name}_{jexc_name}_{suffix}.dat'
    arq_out = open(arq_out_name, 'w')
        
    text = '''# RPA_diag is equation (1) of arxiv:2502.05144, with d/dr <kcv|K^eh|k'c'v'> = 0
# RPA is equation (3) of arxiv:2502.05144, with  <kcv|d K^eh / dr|k'c'v'> = 0
# RPA_diag_plus_Kernel is equation (1) of arxiv:2502.05144 with  <kcv|d K^eh / dr|k'c'v'> = 0'''

    if use_second_derivatives_elph_coeffs == False:
        text += '''\n# Forces units are (eV/ang)\n'''
    else:
        text += '''\n# exc-ph units are (eV/ang^2)\n'''

    arq_out.write(text)
    if verbose:
        print(text)
    header = f'# {"Atom":<5} {"dir":<5}     {"RPA_diag":<25}       {"RPA_diag_offdiag":<25}      {"RPA_diag_plus_Kernel":<25}'
    if verbose:
        print(header)
    arq_out.write(header + '\n')

    for iatom in range(Nat):
        for idir in range(3):
            text = f'{iatom+1:<5} {DIRECTION[idir]:<5}     {F_RPA_diag[iatom, idir]:<25.8f}       {F_RPA[iatom, idir]:<25.8f}      {(F_kernel+F_RPA_diag)[iatom, idir]:<25.8f}'
            if verbose:
                print(text)
            arq_out.write(text + '\n')

    arq_out.close()
    
def print_final_msg(TASKS):
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\n\nFinished calculations at: ", dt_string)

    # timing report
    print("\n\n\n\n")
    TOTAL_TIME = 0.0

    # Set the column widths
    task_column_width = 60
    time_column_width = 10

    print(f"{'Task':<{task_column_width}}{'Time (seconds)':>{time_column_width}}")  # Header
    print("-" * (task_column_width + time_column_width))  # Divider

    for task in TASKS:
        TOTAL_TIME += task[1]
        print(f"{task[0]:<{task_column_width}}{task[1]:>{time_column_width}.2f}")

    print("-" * (task_column_width + time_column_width))  # Divider
    print(f"{'Total':<{task_column_width}}{TOTAL_TIME:>{time_column_width}.2f}")

    print('''
####################################################################

Excited state forces calculation
Developers: Rafael Del Grande, David Strubbe

Please cite: 

@misc{arxivdelgrande2025,
    title={Revisiting ab-initio excited state forces from many-body Green's function formalism: approximations and benchmark}, 
    author={Rafael R. Del Grande and David A. Strubbe},
    year={2025},
    eprint={2502.05144},
    archivePrefix={arXiv},
    primaryClass={cond-mat.mtrl-sci},
    url={https://arxiv.org/abs/2502.05144}, 
}

####################################################################

''')

################################################## code starts here ################################################

if __name__ == "__main__":
    
    # list of tasks and timing for logging
    TASKS = []
    TIMING = []
    
    read_input('forces.inp')
    
    # pass variables from config dictionary to the global variables
    
    iexc = config['iexc']
    jexc = config['jexc']
    factor_head = config['factor_head']
    ncbnds_sum = config['ncbnds_sum']
    nvbnds_sum = config['nvbnds_sum']
    eqp_file = config['eqp_file']
    exciton_file = config['exciton_file']
    el_ph_dir = config['el_ph_dir']
    kernel_file = config['kernel_file']
    calc_modes_basis = config['calc_modes_basis']
    write_DKernel = config['write_DKernel']
    Calculate_Kernel = config['Calculate_Kernel']
    just_RPA_diag = config['just_RPA_diag']
    report_RPA_data = config['report_RPA_data']
    show_imag_part = config['show_imag_part']
    acoutic_sum_rule = config['acoutic_sum_rule']
    use_hermicity_F = config['use_hermicity_F']
    log_k_points = config['log_k_points']
    read_Acvk_pos = config['read_Acvk_pos']
    Acvk_directory = config['Acvk_directory']
    no_renorm_elph = config['no_renorm_elph']
    elph_fine_a_la_bgw = config['elph_fine_a_la_bgw']
    ncbands_co = config['ncbands_co']
    nvbands_co = config['nvbands_co']
    nkpnts_co = config['nkpnts_co']
    write_dK_mat = config['write_dK_mat']
    trust_kpoints_order = config['trust_kpoints_order']
    run_parallel = config['run_parallel']
    num_processes = config['num_processes']
    use_Acvk_single_transition = config['use_Acvk_single_transition']
    dfpt_irreps_list = config['dfpt_irreps_list']
    limit_BSE_sum = config['limit_BSE_sum']
    limit_BSE_sum_up_to_value = config['limit_BSE_sum_up_to_value']
    do_vectorized_sums = config['do_vectorized_sums']
    read_exciton_pairs_file = config['read_exciton_pairs_file']
    hbse_file = config['hbse_file']
    save_elph_coeffs = config['save_elph_coeffs']
    load_elph_coeffs = config['load_elph_coeffs']
    just_save_elph_coeffs = config['just_save_elph_coeffs']
    elph_coeffs_file_to_be_loaded = config["elph_coeffs_file_to_be_loaded"]
    elph_h5_file      = config['elph_h5_file']
    dtmat_file        = config['dtmat_file']
    elph_fine_h5_file = config['elph_fine_h5_file']
    use_second_derivatives_elph_coeffs = config['use_second_derivatives_elph_coeffs']
    
    if run_parallel == True:
        from multiprocessing import Pool
        from functools import partial
        from itertools import islice

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\n\nExecution date: ", dt_string)
    import subprocess

    try:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _hash = subprocess.check_output(["git", "-C", _script_dir, "rev-parse", "--short", "HEAD"], text=True).strip()
        _date = subprocess.check_output(["git", "-C", _script_dir, "log", "-1", "--format=%ad", "--date=short"], text=True).strip()
        CODE_VERSION = f"{_hash} from ({_date})"
    except Exception:
        CODE_VERSION = "not found"

    print("Code version:", CODE_VERSION)
    
    print('''
This code calcultes excited state forces based on Hellman-Feynman theorem 
within the Bethe-Salpeter Hamiltonian. F = - <S|dH/dR|S> where |S> is the excitonic state and H is the BSE Hamiltonian.
The code report forces with the minus sign, so if you want to use 
exciton-phonon matrix elements multiply the results by -1.

More details can be found at: https://github.com/rafaeldgrande/excited_state_forces
and in our paper below:
          ''')
    
    print('''
        
####################################################################

Excited state forces calculation
Developers: Rafael Del Grande, David Strubbe

Please cite: 

@misc{arxivdelgrande2025,
    title={Revisiting ab-initio excited state forces from many-body Green's function formalism: approximations and benchmark}, 
    author={Rafael R. Del Grande and David A. Strubbe},
    year={2025},
    eprint={2502.05144},
    archivePrefix={arXiv},
    primaryClass={cond-mat.mtrl-sci},
    url={https://arxiv.org/abs/2502.05144}, 
}

####################################################################

''')
    

    ######################### RUNNING CODE ##############################

    start_time = time.clock_gettime(0)
    # Getting BSE and MF parameters
    # Reading eigenvecs.h5 file
    time0 = time.clock_gettime(0)
    Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs, BSE_params, MF_params, NQ, Qshift = get_BSE_MF_params()
    time1 = time.clock_gettime(0)
    TASKS.append(['Get parameters from DFT and GWBSE', time1 - time0])

    if NQ == 1:
        Qshift = Qshift[0] # orignal shape is (1,3), now it is (3,)
        
    if use_second_derivatives_elph_coeffs == True:
        print('Using second derivatives of elph coefficients to calculate forces. The units of exc-ph matrix elements in this case will be eV/angstrom**2.')

    read_exciton_pairs(config)

    print("Exciton-ph matrix elements to be computed:")
    exciton_pairs = config['exciton_pairs']
    for exc_pair in exciton_pairs:
        print(f" <{exc_pair[0]} | dH | {exc_pair[1]}>")
        
    # definning excitons to be loaded
    excitons_to_be_loaded = {num for pair in exciton_pairs for num in pair}
    excitons_to_be_loaded = sorted(excitons_to_be_loaded)
    # print(f"Excitons to be loaded: {excitons_to_be_loaded}")
    
    # loading excitons coefficients
    time0 = time.clock_gettime(0)
    print(f"Loading exciton coefficients from file {exciton_file}")
    Exciton_coeffs = load_excitons_coefficients(exciton_file, excitons_to_be_loaded)
    # shape (Exciton_coeffs) is (Loaded Nexc, Nkpoints_BSE, Ncbnds, Nvbnds)
    time1 = time.clock_gettime(0)
    print(f"Finished loading exciton coefficients")
    TASKS.append(['Loading exciton coefficients', time1 - time0])
    
    # defining exciton pairs indexes to be consistent with exciton coefficients array
    exciton_pairs_indexes = []
    for pair in exciton_pairs:
        iexc, jexc = pair
        id_i, id_j = excitons_to_be_loaded.index(iexc), excitons_to_be_loaded.index(jexc)
        exciton_pairs_indexes.append((id_i, id_j))
    # print('exciton_pairs_indexes:', exciton_pairs_indexes)

    # getting info from eqp.dat in the fine grid (from absorption calculation)
    time0 = time.clock_gettime(0)
    Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, BSE_params)
    # Shape Eqp_val = (Nkpoints_BSE, Nvbnds)
    # Shape Eqp_cond = (Nkpoints_BSE, Ncbnds)
    time1 = time.clock_gettime(0)
    TASKS.append(['Reading QP and DFT energy levels', time1 - time0])

    # load pre-interpolated fine-grid el-ph from HDF5 (produced by interpolate_elph_bgw.py)
    # elph_fine_cond shape: (Nq, Nmodes, Nk_fi, Nc_fi, Nc_fi)
    # elph_fine_val  shape: (Nq, Nmodes, Nk_fi, Nv_fi, Nv_fi)
    time0 = time.clock_gettime(0)
    print(f'\nLoading fine-grid el-ph from {elph_fine_h5_file}')
    with h5py.File(elph_fine_h5_file, 'r') as fh:
        for _key in ('elph_fine_cond_mode', 'elph_fine_val_mode',
                     'elph_fine_cond_cart', 'elph_fine_val_cart',
                     'Kpoints_in_elph_file', 'phonon_modes/eigenvectors',
                     'phonon_modes/frequencies'):
            if _key not in fh:
                raise KeyError(
                    f"'{_key}' not found in {elph_fine_h5_file}. "
                    f"Re-run interpolate_elph_bgw.py to regenerate the file.")
        elph_cond_mode   = fh['elph_fine_cond_mode'][0].astype(np.complex64)  # (Nmodes, Nk_fi, Nc_fi, Nc_fi)
        elph_val_mode    = fh['elph_fine_val_mode'][0].astype(np.complex64)   # (Nmodes, Nk_fi, Nv_fi, Nv_fi)
        elph_cond_cart   = fh['elph_fine_cond_cart'][0].astype(np.complex64)  # (Npert,  Nk_fi, Nc_fi, Nc_fi)
        elph_val_cart    = fh['elph_fine_val_cart'][0].astype(np.complex64)   # (Npert,  Nk_fi, Nv_fi, Nv_fi)
        Kpoints_in_elph_file = fh['Kpoints_in_elph_file'][:]                  # (Nk_fi, 3) crystal coords
        Displacements        = fh['phonon_modes/eigenvectors'][0]               # (Nmodes, Nat, 3), iq=0
        phonon_frequencies   = fh['phonon_modes/frequencies'][0]               # (Nmodes,) in cm^-1
    print(f'  elph_cond_mode shape: {elph_cond_mode.shape}  (Nmodes, Nk_fi, Nc_fi, Nc_fi)')
    print(f'  elph_val_mode  shape: {elph_val_mode.shape}  (Nmodes, Nk_fi, Nv_fi, Nv_fi)')
    print(f'  elph_cond_cart shape: {elph_cond_cart.shape}  (Npert,  Nk_fi, Nc_fi, Nc_fi)')
    print(f'  elph_val_cart  shape: {elph_val_cart.shape}  (Npert,  Nk_fi, Nv_fi, Nv_fi)')
    print(f'  Displacements:        {Displacements.shape}  (Nmodes, Nat, 3)')
    time1 = time.clock_gettime(0)
    TASKS.append(['Loading fine-grid ELPH from h5 (interpolate_elph_bgw output)', time1 - time0])
    
    Nmodes = elph_cond_mode.shape[0]   # number of phonon modes
    Npert  = elph_cond_cart.shape[0]   # = 3*Nat Cartesian perturbations
    MF_params.Nmodes = Nmodes

    # ensure shape (Nk_fi, 3) — guard against transposed save
    if Kpoints_in_elph_file.shape[-1] != 3:
        Kpoints_in_elph_file = Kpoints_in_elph_file.T
    Nkpoints_DFPT = len(Kpoints_in_elph_file)

    if verbosity == 'high':
        print(f'\nFine-grid ELPH k-points: {Nkpoints_DFPT} k-points (crystal coords)')

    # k-points are already in crystal (fractional) coords — just wrap to [0, 1)
    time0 = time.clock_gettime(0)
    Kpoints_in_elph_file_frac = np.array([
        [correct_comp_vector(c) for c in k]
        for k in Kpoints_in_elph_file
    ])

    if verbosity == 'high':
        print(f'Matching BSE k-points ({Nkpoints_BSE}) to fine ELPH k-points ({Nkpoints_DFPT}) ...')
    ikBSE_to_ikDFPT = translate_bse_to_dfpt_k_points()
    if verbosity == 'high':
        n_missing = ikBSE_to_ikDFPT.count(-1)
        print(f'  Matched {Nkpoints_BSE - n_missing}/{Nkpoints_BSE} BSE k-points'
              + (f' ({n_missing} missing)' if n_missing else ' — all matched'))

    # renormalize ELPH coefficients (both bases)
    time0 = time.clock_gettime(0)
    def renormalize_elph_coeffs(elph, Eqp, Edft):
        # elph: (Npert, nk, nb, nb),  Eqp/Edft: (nk, nb)
        dEqp  = Eqp[:, :, None] - Eqp[:, None, :]    # (nk, nb, nb)
        dEdft = Edft[:, :, None] - Edft[:, None, :]   # (nk, nb, nb)
        mask  = np.abs(dEdft) > TOL_DEG
        ratio = np.where(mask, dEqp / np.where(mask, dEdft, 1.0), 1.0)
        elph *= ratio[None, ...]
        return elph

    if renormalize_elph_coeffs == True:
        print('Renormalizing ELPH coefficients using QP and DFT energy levels')
        elph_cond_mode = renormalize_elph_coeffs(elph_cond_mode, Eqp_cond, Edft_cond)
        elph_val_mode  = renormalize_elph_coeffs(elph_val_mode,  Eqp_val,  Edft_val)
        elph_cond_cart = renormalize_elph_coeffs(elph_cond_cart, Eqp_cond, Edft_cond)
        elph_val_cart  = renormalize_elph_coeffs(elph_val_cart,  Eqp_val,  Edft_val)
    else:
        print('Not renormalizing ELPH coefficients using QP and DFT energy levels')
    TASKS.append(['Renormalizing ELPH coefficients using QP and DFT energy levels', time.clock_gettime(0) - time0])

    def _expand_elph(elph_cond, elph_val, Npert_dim, label):
        """Expand elph to augmented (Npert, Nk, Nc, Nv, Nc, Nv) matrices."""
        Gc      = np.zeros((Npert_dim, Nkpoints_BSE, Ncbnds, Nvbnds, Ncbnds, Nvbnds), dtype=np.complex64)
        Gv      = np.zeros_like(Gc)
        Gc_diag = np.zeros((Npert_dim, Nkpoints_BSE, Ncbnds, Nvbnds), dtype=np.complex64)
        Gv_diag = np.zeros_like(Gc_diag)
        if verbosity == 'high':
            print(f'  {label}: cond {elph_cond.shape} → {Gc.shape}, val {elph_val.shape} → {Gv.shape}')
        for iv in range(Nvbnds):
            Gc[:, :, :, iv, :, iv] = elph_cond
        for ic in range(Ncbnds):
            Gv[:, :, ic, :, ic, :] = elph_val
        Gc_diag[:] = np.diagonal(elph_cond, axis1=2, axis2=3)[:, :, :, np.newaxis]
        Gv_diag[:] = np.diagonal(elph_val,  axis1=2, axis2=3)[:, :, np.newaxis, :]
        return Gc, Gv, Gc_diag, Gv_diag

    time0 = time.clock_gettime(0)
    print("\nExpanding ELPH matrices for vectorized multiplication")

    Gc_mode, Gv_mode, Gc_mode_diag, Gv_mode_diag = _expand_elph(elph_cond_mode, elph_val_mode, Nmodes, 'phonon-mode')
    Gc_cart, Gv_cart, Gc_cart_diag, Gv_cart_diag = _expand_elph(elph_cond_cart, elph_val_cart, Npert,  'Cartesian')

    # apply Q shift on valence states (finite-momentum BSE) — print once
    Gv_mode = apply_Qshift_on_valence_states(Qshift, Gv_mode, Kpoints_in_elph_file_frac, verbose=True)
    Gv_cart = apply_Qshift_on_valence_states(Qshift, Gv_cart, Kpoints_in_elph_file_frac, verbose=False)

    time1 = time.clock_gettime(0)
    TASKS.append(['ELPH matrices expansion (for vectorized multiplication)', time1 - time0])

    dRPA_dr_mode_mat      = Gc_mode - Gv_mode
    dRPA_dr_mode_diag_mat = Gc_mode_diag - Gv_mode_diag
    dRPA_dr_cart_mat      = Gc_cart - Gv_cart
    dRPA_dr_cart_diag_mat = Gc_cart_diag - Gv_cart_diag
    
    
    # Loading Kernel matrix elements
    if Calculate_Kernel == True:
        print(f"I will obtain the kernel matrix in the fine grid using the {hbse_file} and eqp.dat files produced by absorption")
        time0 = time.clock_gettime(0)
        
        # reconstruct kernel from hbse.h5 and eqp.dat in the fine grid
        # teste = load_hbse_matrix(hbse_file, Nkpoints_BSE, Ncbnds, Nvbnds)
        kernel_matrix = get_kernel_from_hbse(hbse_file, Eqp_cond, Eqp_val) # units eV
        print('Finished loading kernel matrix elements from hbse file\n')
        time1 = time.clock_gettime(0)
        TASKS.append(['Loading kernel matrix elements', time1 - time0])
        
        time0 = time.clock_gettime(0)
        print("Calculating d/dr_imode <kcv|K|kc'v'> matrix elements. Equation (29) of arxiv:2502.05144")
        DKernel_dr_imode_mat = calc_Dkernel_new(kernel_matrix, elph_cond_mode, elph_val_mode, Eqp_cond, Eqp_val, vectorized=do_vectorized_sums) # units ry/bohr
        time1 = time.clock_gettime(0)
        TASKS.append(['Calculating d/dr_imode <kcv|K|kc\'v\'> matrix elements', time1 - time0])

    def f_cart_to_atomic_forces(f_cart_pert):
        """Convert Cartesian-basis forces (3*Nat,) [Ry/bohr] to atomic (Nat,3) [eV/ang]."""
        return -np.array(f_cart_pert).reshape(Nat, 3) * Ry2eV / bohr2A

    ####### Computing forces for each exciton pair — both bases ##########

    forces_ph_RPA      = []
    forces_ph_RPA_diag = []
    forces_ph_Kernel   = []
    forces_ca_RPA      = []
    forces_ca_RPA_diag = []
    forces_ca_Kernel   = []

    time_calc_RPA_diag, time_calc_RPA, time_calc_Kernel = 0.0, 0.0, 0.0

    _unit = -Ry2eV / bohr2A   # Ry/bohr → eV/ang, with sign from F = -dH/dR

    def process_exciton_pair(exc_pair):
        Akcv, Bkcv = Exciton_coeffs[exc_pair[0]-1], Exciton_coeffs[exc_pair[1]-1]

        # ── phonon-mode basis: keep raw mode forces (Nmodes,) in eV/ang ──
        time0 = time.clock_gettime(0)
        f_ph_rpa      = compute_A_dRPA_dr_imode_B(Akcv, Bkcv, dRPA_dr_mode_mat, elph_cond_mode, elph_val_mode, vectorized=do_vectorized_sums) * _unit
        f_ph_rpa_diag = compute_A_dRPAdiag_dr_imode_B(Akcv, Bkcv, dRPA_dr_mode_diag_mat, elph_cond_mode, elph_val_mode, vectorized=do_vectorized_sums) * _unit
        time_rpa = time.clock_gettime(0) - time0

        # ── Cartesian basis: reshape (3*Nat,) → (Nat, 3) ──
        time0 = time.clock_gettime(0)
        f_ca_rpa      = f_cart_to_atomic_forces(compute_A_dRPA_dr_imode_B(Akcv, Bkcv, dRPA_dr_cart_mat, elph_cond_cart, elph_val_cart, vectorized=do_vectorized_sums))
        f_ca_rpa_diag = f_cart_to_atomic_forces(compute_A_dRPAdiag_dr_imode_B(Akcv, Bkcv, dRPA_dr_cart_diag_mat, elph_cond_cart, elph_val_cart, vectorized=do_vectorized_sums))
        time_rpa_diag = time.clock_gettime(0) - time0

        f_ph_kernel = f_ca_kernel = None
        time_kernel = 0.0
        if Calculate_Kernel == True:
            time0 = time.clock_gettime(0)
            f_ph_kernel = compute_A_dKernel_dr_imode_B(Akcv, Bkcv, DKernel_dr_imode_mat, vectorized=do_vectorized_sums) * _unit
            time_kernel = time.clock_gettime(0) - time0

        return (exc_pair,
                f_ph_rpa, f_ph_rpa_diag, f_ph_kernel,
                f_ca_rpa, f_ca_rpa_diag, f_ca_kernel,
                time_rpa, time_rpa_diag, time_kernel)

    def _collect(result):
        (exc_pair,
         rpa_ph, rpa_diag_ph, kernel_ph,
         rpa_ca, rpa_diag_ca, kernel_ca,
         time_rpa, time_rpa_diag, time_kernel) = result
        forces_ph_RPA.append(rpa_ph);        forces_ph_RPA_diag.append(rpa_diag_ph)
        forces_ca_RPA.append(rpa_ca);        forces_ca_RPA_diag.append(rpa_diag_ca)
        if kernel_ph is not None: forces_ph_Kernel.append(kernel_ph)
        if kernel_ca is not None: forces_ca_Kernel.append(kernel_ca)
        return time_rpa, time_rpa_diag, time_kernel

    total_pairs = len(exciton_pairs)
    if run_parallel == False:
        print("\n\n################################# Running in serial ################################")
        print("Total exciton-phonon matrix elements to be calculated: ", total_pairs)
        for i_pair, exc_pair in enumerate(exciton_pairs):
            if i_pair > 0 and (i_pair == 1 or i_pair == 5 or i_pair % 10 == 0 or i_pair == total_pairs - 1):
                print(f"Progress: {i_pair/total_pairs*100:.1f}% ({i_pair}/{total_pairs})")
            tr, trd, tk = _collect(process_exciton_pair(exc_pair))
            time_calc_RPA += tr;  time_calc_RPA_diag += trd;  time_calc_Kernel += tk
    else:
        print(f"\n\n################################# Running in parallel with {num_processes} processes #################################")
        if num_processes == 1:
            print("Warning: Running in parallel with just one process. No speedup will be achieved! Change num_processes in forces.inp file.")
        print("Total exciton-phonon matrix elements to be calculated: ", total_pairs)
        import multiprocessing
        ctx = multiprocessing.get_context('fork')
        with ctx.Pool(processes=num_processes) as pool:
            results = []
            for i_pair, result in enumerate(pool.imap(process_exciton_pair, exciton_pairs)):
                results.append(result)
                if i_pair > 0 and (i_pair == 1 or i_pair == 5 or i_pair % 10 == 0 or i_pair == total_pairs - 1):
                    print(f"Progress: {i_pair/total_pairs*100:.1f}% ({i_pair}/{total_pairs})")
        for result in results:
            tr, trd, tk = _collect(result)
            time_calc_RPA += tr;  time_calc_RPA_diag += trd;  time_calc_Kernel += tk

    TASKS.append(['Calculating forces with RPA_diag', time_calc_RPA_diag])
    TASKS.append(['Calculating forces with RPA', time_calc_RPA])
    TASKS.append(['Calculating forces with Kernel part', time_calc_Kernel])

    forces_ph_RPA      = np.array(forces_ph_RPA)
    forces_ph_RPA_diag = np.array(forces_ph_RPA_diag)
    forces_ca_RPA      = np.array(forces_ca_RPA)
    forces_ca_RPA_diag = np.array(forces_ca_RPA_diag)
    if Calculate_Kernel:
        forces_ph_Kernel = np.array(forces_ph_Kernel)

    for i, exc_pair in enumerate(exciton_pairs):
        iexc, jexc = exc_pair
        verbose = len(exciton_pairs) == 1
        report_forces_ph(iexc-1, jexc-1,
                         forces_ph_RPA_diag[i], forces_ph_RPA[i],
                         forces_ph_Kernel[i] if Calculate_Kernel else None,
                         phonon_frequencies, verbose=verbose)
        report_forces(iexc-1, jexc-1, forces_ca_RPA_diag[i], forces_ca_RPA[i],
                      None, suffix='cart', verbose=verbose)
        
    
    print_final_msg(TASKS)

