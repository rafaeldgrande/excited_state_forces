
verbosity = 'high'

# excited state forces modules
from modules_to_import import *
from excited_forces_config import *
from bgw_interface_m import *
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
    Calculate_Kernel = config['Calculate_Kernel']
    just_RPA_diag = config['just_RPA_diag']
    acoutic_sum_rule = config['acoutic_sum_rule']
    use_hermicity_F = config['use_hermicity_F']
    log_k_points = config['log_k_points']
    read_Acvk_pos = config['read_Acvk_pos']
    Acvk_directory = config['Acvk_directory']
    no_renorm_elph = config['no_renorm_elph']
    write_dK_mat = config['write_dK_mat']
    trust_kpoints_order = config['trust_kpoints_order']
    run_parallel = config['run_parallel']
    num_processes = config['num_processes']
    dfpt_irreps_list = config['dfpt_irreps_list']
    do_vectorized_sums = config['do_vectorized_sums']
    read_exciton_pairs_file = config['read_exciton_pairs_file']
    hbse_file = config['hbse_file']
    elph_fine_h5_file = config['elph_fine_h5_file']
    use_second_derivatives_elph_coeffs = config['use_second_derivatives_elph_coeffs']
    save_forces_h5    = config['save_forces_h5']
    forces_h5_file    = config['forces_h5_file']
    finite_q_phonon   = config['finite_q_phonon']
    exciton_A_file = config['exciton_A_file']
    exciton_B_file = config['exciton_B_file']
    use_inv_symm_q_grid = config['use_inv_symm_q_grid']
    
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

    if finite_q_phonon:
        print(f'\nFinite-q phonon mode: using {exciton_A_file} for BSE parameters')
        config['exciton_file'] = exciton_A_file

    start_time = time.clock_gettime(0)
    # Getting BSE and MF parameters
    # Reading eigenvecs.h5 file
    time0 = time.clock_gettime(0)
    Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs, BSE_params, MF_params, NQ, Qshift = get_BSE_MF_params()
    time1 = time.clock_gettime(0)
    TASKS.append(['Get parameters from DFT and GWBSE', time1 - time0])

    if NQ == 1:
        Qshift = Qshift[0] # orignal shape is (1,3), now it is (3,)

    q_phonon = np.zeros(3)   # phonon momentum; set below when finite_q_phonon=True

    if finite_q_phonon:
        Q_A = Qshift   # momentum of exciton A (shape (3,))
        with h5py.File(exciton_B_file, 'r') as _fh_b:
            _Qshift_B = _fh_b['/exciton_header/kpoints/exciton_Q_shifts'][()]
        Q_B = _Qshift_B[0] if _Qshift_B.ndim == 2 else _Qshift_B
        _q_raw = Q_B - Q_A
        q_phonon = _q_raw - np.round(_q_raw)   # fold to [-0.5, 0.5) first BZ
        print(f'  Q_A (exciton A momentum) = {Q_A}')
        print(f'  Q_B (exciton B momentum) = {Q_B}')
        print(f'  phonon q = Q_B - Q_A     = {q_phonon} (folded to first BZ)')
        
    if use_second_derivatives_elph_coeffs == True:
        print('Using second derivatives of elph coefficients to calculate forces. The units of exc-ph matrix elements in this case will be eV/angstrom**2.')

    read_exciton_pairs(config)

    
    exciton_pairs = config['exciton_pairs']
    print("Total exciton-ph matrix elements to be computed:", len(exciton_pairs))
    for i_pair in range(min(len(exciton_pairs), 5)):
        exc_pair = exciton_pairs[i_pair]
        print(f" <{exc_pair[0]} | dH | {exc_pair[1]}>")
    if len(exciton_pairs) > 5:
        print(f" ... and {len(exciton_pairs) - 5} terms more")
        
    # loading exciton coefficients
    time0 = time.clock_gettime(0)
    if finite_q_phonon:
        # A and B come from separate eigenvectors files
        excitons_A_to_load = sorted({pair[0] for pair in exciton_pairs})
        excitons_B_to_load = sorted({pair[1] for pair in exciton_pairs})
        print(f"Loading exciton A coefficients from {exciton_A_file}")
        Exciton_coeffs_A, exciton_eigenvalues_A = load_excitons_coefficients(
            exciton_A_file, excitons_A_to_load)
        print(f"Loading exciton B coefficients from {exciton_B_file}")
        Exciton_coeffs_B, exciton_eigenvalues_B = load_excitons_coefficients(
            exciton_B_file, excitons_B_to_load)
        exciton_eigenvalues = np.concatenate([exciton_eigenvalues_A, exciton_eigenvalues_B])
        # keep a reference list consistent with the non-finite_q path for the h5 save
        excitons_to_be_loaded = excitons_A_to_load  # used only in save_exc_forces_h5 label
    else:
        excitons_to_be_loaded = sorted({num for pair in exciton_pairs for num in pair})
        print(f"Loading exciton coefficients from file {exciton_file}")
        Exciton_coeffs, exciton_eigenvalues = load_excitons_coefficients(
            exciton_file, excitons_to_be_loaded)
        # Exciton_coeffs shape: (Loaded Nexc, Nkpoints_BSE, Ncbnds, Nvbnds)
    time1 = time.clock_gettime(0)
    print(f"Finished loading exciton coefficients")
    TASKS.append(['Loading exciton coefficients', time1 - time0])

    # QP data: will be populated from elph.h5 if present, else read from eqp.dat below
    QP_rescaling_cond = None
    QP_rescaling_val  = None
    Eqp_cond          = None
    Eqp_val           = None
    Edft_cond         = None
    Edft_val          = None

    # load pre-interpolated fine-grid el-ph from HDF5 (produced by interpolate_elph_bgw.py)
    # elph_fine_cond shape: (Nq, Nmodes, Nk_fi, Nc_fi, Nc_fi)
    # elph_fine_val  shape: (Nq, Nmodes, Nk_fi, Nv_fi, Nv_fi)
    time0 = time.clock_gettime(0)
    print(f'\nLoading fine-grid el-ph from {elph_fine_h5_file}')
    print(  '  This file contains interpolated electron-phonon matrix elements')
    print(  '  <n,k+q|dV(q)|m,k> on the fine k-grid (produced by interpolate_elph_bgw.py).')
    
    with h5py.File(elph_fine_h5_file, 'r') as fh:
        # print file-level metadata if available
        _attrs = {k: fh.attrs[k] for k in fh.attrs}
        if _attrs:
            print('  File attributes:')
            for _k, _v in _attrs.items():
                print(f'    {_k} = {_v}')
        _required = ('elph_fine_cond_mode', 'elph_fine_val_mode',
                     'elph_fine_cond_cart', 'elph_fine_val_cart',
                     'Kpoints_in_elph_file', 'phonon_modes/eigenvectors',
                     'phonon_modes/frequencies')
        if finite_q_phonon:
            _required = _required + ('qpoints_crystal',)
        for _key in _required:
            if _key not in fh:
                raise KeyError(
                    f"'{_key}' not found in {elph_fine_h5_file}. "
                    f"Re-run interpolate_elph_bgw.py to regenerate the file.")

        # Track whether we need to apply the g(q) → g(-q) inversion-symmetry transform
        _used_minus_q = False

        if finite_q_phonon:
            qpoints_crystal = fh['qpoints_crystal'][:]   # (Nq, 3) fractional coords
            iq_phonon = find_kpoint(q_phonon, qpoints_crystal)
            if iq_phonon == -1 and use_inv_symm_q_grid:
                # Phonons at -q satisfy g(-q)_{nm} = conj(g(q)_{mn}), so try -q_phonon
                iq_phonon = find_kpoint(-q_phonon, qpoints_crystal)
                if iq_phonon != -1:
                    _used_minus_q = True
                    print(f'  q = {q_phonon} not found; using -q = {-q_phonon} '
                          f'(iq={iq_phonon}) via inversion symmetry.')
            if iq_phonon == -1:
                print(f'\nERROR: phonon q = {q_phonon} (= Q_B - Q_A) was NOT found in '
                      f'{elph_fine_h5_file} (Nq = {len(qpoints_crystal)} q-points).')
                if use_inv_symm_q_grid:
                    print(f'  Also tried -q = {-q_phonon}: not found.')
                print('Q-points found in the file:')
                for _iq, _q in enumerate(qpoints_crystal):
                    print(f'  iq={_iq}: {_q}')
                print('Please re-run interpolate_elph_bgw.py including this q-point.')
                import sys; sys.exit(1)
            if not _used_minus_q:
                print(f'  Found phonon q at index iq = {iq_phonon} in elph_fine.h5')
        else:
            iq_phonon = 0

        elph_cond_mode   = fh['elph_fine_cond_mode'][iq_phonon].astype(np.complex64)  # (Nmodes, Nk_fi, Nc_fi, Nc_fi)
        elph_val_mode    = fh['elph_fine_val_mode'][iq_phonon].astype(np.complex64)   # (Nmodes, Nk_fi, Nv_fi, Nv_fi)
        elph_cond_cart   = fh['elph_fine_cond_cart'][iq_phonon].astype(np.complex64)  # (Npert,  Nk_fi, Nc_fi, Nc_fi)
        elph_val_cart    = fh['elph_fine_val_cart'][iq_phonon].astype(np.complex64)   # (Npert,  Nk_fi, Nv_fi, Nv_fi)

        if _used_minus_q:
            # g(-q)_{nm} = conj(g(q)_{mn}) — conjugate and swap the two band indices
            elph_cond_mode = np.conj(elph_cond_mode).transpose(0, 1, 3, 2)
            elph_val_mode  = np.conj(elph_val_mode).transpose(0, 1, 3, 2)
            elph_cond_cart = np.conj(elph_cond_cart).transpose(0, 1, 3, 2)
            elph_val_cart  = np.conj(elph_val_cart).transpose(0, 1, 3, 2)
        Kpoints_in_elph_file = fh['Kpoints_in_elph_file'][:]                          # (Nk_fi, 3) crystal coords

        # phonon_modes/* may have a different q-point ordering than elph_fine datasets.
        # Use qpoints_crystal (fractional) for matching — compare directly with q_phonon.
        _qpts_cryst = fh['qpoints_crystal'][:]                    # (Nq, 3) crystal coords
        _q_lookup = (-q_phonon if _used_minus_q else q_phonon)
        _iq_modes = next(
            (i for i, qm in enumerate(_qpts_cryst)
             if np.linalg.norm((_q_lookup - qm) - np.round(_q_lookup - qm)) < 1e-5), -1)
        if _iq_modes == -1:
            print(f'  WARNING: phonon q = {_q_lookup} not found in '
                  f'qpoints_crystal of {elph_fine_h5_file}.')
            print(f'  Falling back to Gamma (index 0) for phonon eigenvectors/frequencies.')
            print(f'  NOTE: if matdyn.x was not run at this q, g_mode at iq={iq_phonon} '
                  f'is zero in elph.h5 and ph-mode forces will be zero.')
            _iq_modes = 0
        else:
            print(f'  phonon eigenvectors: using phonon_modes index {_iq_modes} '
                  f'(q_crystal = {_qpts_cryst[_iq_modes]})')

        Displacements        = fh['phonon_modes/eigenvectors'][_iq_modes]              # (Nmodes, Nat, 3)
        phonon_frequencies   = fh['phonon_modes/frequencies'][_iq_modes]              # (Nmodes,) in cm^-1

        if 'QP_rescaling_matrix_cond' in fh:
            QP_rescaling_cond = fh['QP_rescaling_matrix_cond'][:]   # (Nk_fi, Nc_fi, Nc_fi)
            QP_rescaling_val  = fh['QP_rescaling_matrix_val'][:]    # (Nk_fi, Nv_fi, Nv_fi)
            Eqp_cond  = fh['Eqp_cond'][:]                           # (Nk_fi, Nc_fi)
            Eqp_val   = fh['Eqp_val'][:]                            # (Nk_fi, Nv_fi)
            Edft_cond = fh['Edft_cond'][:] if 'Edft_cond' in fh else None
            Edft_val  = fh['Edft_val'][:] if 'Edft_val' in fh else None
            print(f'  Loaded QP rescaling matrices and energies from {elph_fine_h5_file}')

    _Nm, _Nk, _Nc = elph_cond_mode.shape[0], elph_cond_mode.shape[1], elph_cond_mode.shape[2]
    _Nv  = elph_val_mode.shape[2]
    _Np  = elph_cond_cart.shape[0]
    _Nat = Displacements.shape[1]
    print(f'\n  Loaded arrays (iq = {iq_phonon}):')
    print(f'    Nmodes = {_Nm}   Nk_fi = {_Nk}   Nc_fi = {_Nc}   Nv_fi = {_Nv}   Npert = {_Np}   Nat = {_Nat}')
    print(f'    elph_cond_mode : {elph_cond_mode.shape}  (Nmodes, Nk_fi, Nc_fi, Nc_fi)')
    print(f'    elph_val_mode  : {elph_val_mode.shape}  (Nmodes, Nk_fi, Nv_fi, Nv_fi)')
    print(f'    elph_cond_cart : {elph_cond_cart.shape}  (Npert,  Nk_fi, Nc_fi, Nc_fi)')
    print(f'    elph_val_cart  : {elph_val_cart.shape}  (Npert,  Nk_fi, Nv_fi, Nv_fi)')
    print(f'    Displacements  : {Displacements.shape}  (Nmodes, Nat, 3)')
    print(f'    phonon freqs   : {phonon_frequencies.shape}  min={phonon_frequencies.min():.2f}'
          f'  max={phonon_frequencies.max():.2f} cm⁻¹')
    time1 = time.clock_gettime(0)
    TASKS.append(['Loading fine-grid ELPH from h5 (interpolate_elph_bgw output)', time1 - time0])

    # QP energies: use data from elph.h5 if present, else read eqp.dat
    time0 = time.clock_gettime(0)
    if Eqp_cond is not None:
        print(f'Using QP energies from {elph_fine_h5_file} (skipping {eqp_file})')
    else:
        Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, BSE_params)
    time1 = time.clock_gettime(0)
    TASKS.append(['Reading QP and DFT energy levels', time1 - time0])

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
    def renormalize_elph_coeffs(elph, Eqp, Edft, ratio=None):
        # elph: (Npert_or_Nmodes, nk, nb, nb)
        # If ratio (nk, nb, nb) is pre-computed, use it directly; else compute from Eqp/Edft.
        if ratio is None:
            dEqp  = Eqp[:, :, None] - Eqp[:, None, :]    # (nk, nb, nb)
            dEdft = Edft[:, :, None] - Edft[:, None, :]
            mask  = np.abs(dEdft) > TOL_DEG
            ratio = np.where(mask, dEqp / np.where(mask, dEdft, 1.0), 1.0)
        elph *= ratio[None, ...]
        return elph

    if not no_renorm_elph:
        print('Renormalizing ELPH coefficients using QP and DFT energy levels')
        elph_cond_mode = renormalize_elph_coeffs(elph_cond_mode, Eqp_cond, Edft_cond, QP_rescaling_cond)
        elph_val_mode  = renormalize_elph_coeffs(elph_val_mode,  Eqp_val,  Edft_val,  QP_rescaling_val)
        elph_cond_cart = renormalize_elph_coeffs(elph_cond_cart, Eqp_cond, Edft_cond, QP_rescaling_cond)
        elph_val_cart  = renormalize_elph_coeffs(elph_val_cart,  Eqp_val,  Edft_val,  QP_rescaling_val)
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

    def save_exc_forces_h5(out_path):
        """Write all forces + run metadata to an HDF5 file."""
        f_ph_kernel_arr = np.array(forces_ph_Kernel) if Calculate_Kernel else np.zeros_like(forces_ph_RPA_diag)
        f_ca_kernel_arr = np.zeros_like(forces_ca_RPA_diag)  # kernel in Cartesian not computed separately

        pairs_arr = np.array(exciton_pairs, dtype=np.int32)

        with h5py.File(out_path, 'w') as h5:
            # ── forces ──────────────────────────────────────────────────
            grp_ph = h5.require_group('forces/ph')
            grp_ph.create_dataset('RPA_diag', data=np.real(forces_ph_RPA_diag))
            grp_ph['RPA_diag'].attrs['axes'] = '(Npairs, Nmodes)'
            grp_ph['RPA_diag'].attrs['units'] = 'eV/ang^2' if use_second_derivatives_elph_coeffs else 'eV/ang'
            grp_ph.create_dataset('RPA', data=np.real(forces_ph_RPA))
            grp_ph['RPA'].attrs['axes'] = '(Npairs, Nmodes)'
            grp_ph['RPA'].attrs['units'] = 'eV/ang^2' if use_second_derivatives_elph_coeffs else 'eV/ang'
            grp_ph.create_dataset('RPA_diag_plus_Kernel', data=np.real(f_ph_kernel_arr + forces_ph_RPA_diag))
            grp_ph['RPA_diag_plus_Kernel'].attrs['axes'] = '(Npairs, Nmodes)'
            grp_ph['RPA_diag_plus_Kernel'].attrs['units'] = 'eV/ang^2' if use_second_derivatives_elph_coeffs else 'eV/ang'

            grp_ca = h5.require_group('forces/cart')
            grp_ca.create_dataset('RPA_diag', data=np.real(forces_ca_RPA_diag))
            grp_ca['RPA_diag'].attrs['axes'] = '(Npairs, Nat, 3)'
            grp_ca['RPA_diag'].attrs['units'] = 'eV/ang^2' if use_second_derivatives_elph_coeffs else 'eV/ang'
            grp_ca.create_dataset('RPA', data=np.real(forces_ca_RPA))
            grp_ca['RPA'].attrs['axes'] = '(Npairs, Nat, 3)'
            grp_ca['RPA'].attrs['units'] = 'eV/ang^2' if use_second_derivatives_elph_coeffs else 'eV/ang'
            grp_ca.create_dataset('RPA_diag_plus_Kernel', data=np.real(f_ca_kernel_arr + forces_ca_RPA_diag))
            grp_ca['RPA_diag_plus_Kernel'].attrs['axes'] = '(Npairs, Nat, 3)'
            grp_ca['RPA_diag_plus_Kernel'].attrs['units'] = 'eV/ang^2' if use_second_derivatives_elph_coeffs else 'eV/ang'

            # ── exciton pair labels ──────────────────────────────────────
            h5.create_dataset('exciton_pairs', data=pairs_arr)
            h5['exciton_pairs'].attrs['description'] = 'Exciton pair (iexc, jexc) for each row in forces arrays (1-based)'

            # ── system parameters ────────────────────────────────────────
            grp_sys = h5.require_group('system')
            grp_sys.attrs['Nkpoints'] = Nkpoints_BSE
            grp_sys.attrs['Ncbnds']   = Ncbnds
            grp_sys.attrs['Nvbnds']   = Nvbnds
            grp_sys.attrs['Nval']     = Nval
            grp_sys.attrs['Nmodes']   = Nmodes
            grp_sys.attrs['Nat']      = Nat
            grp_sys.create_dataset('kpoints_bse', data=Kpoints_BSE)
            grp_sys['kpoints_bse'].attrs['axes']  = '(Nkpoints, 3)'
            grp_sys['kpoints_bse'].attrs['units'] = 'crystal (fractional) coordinates'
            grp_sys.create_dataset('phonon_frequencies', data=phonon_frequencies)
            grp_sys['phonon_frequencies'].attrs['units'] = 'cm^-1'
            # exciton energies for the unique excitons loaded
            grp_sys.create_dataset('exciton_energies', data=exciton_eigenvalues)
            grp_sys['exciton_energies'].attrs['description'] = (
                'Eigenvalues for excitons listed in excitons_to_be_loaded (1-based indices: '
                + str(list(excitons_to_be_loaded)) + ')')
            grp_sys['exciton_energies'].attrs['units'] = 'eV'
            grp_sys.create_dataset('excitons_loaded', data=np.array(list(excitons_to_be_loaded), dtype=np.int32))
            grp_sys['excitons_loaded'].attrs['description'] = '1-based exciton indices loaded'

            # Dense energy arrays: index i → exciton i+1 (eV); 0.0 for not-loaded slots
            if finite_q_phonon:
                _max_A = max(excitons_A_to_load)
                _max_B = max(excitons_B_to_load)
                _en_A  = np.zeros(_max_A)
                _en_B  = np.zeros(_max_B)
                for _k, _idx in enumerate(sorted(excitons_A_to_load)):
                    _en_A[_idx - 1] = exciton_eigenvalues_A[_k]
                for _k, _idx in enumerate(sorted(excitons_B_to_load)):
                    _en_B[_idx - 1] = exciton_eigenvalues_B[_k]
            else:
                _sorted_exc = sorted(excitons_to_be_loaded)
                _en_A = np.zeros(_sorted_exc[-1])
                for _k, _idx in enumerate(_sorted_exc):
                    _en_A[_idx - 1] = exciton_eigenvalues[_k]
                _en_B = _en_A

            grp_sys.create_dataset('exc_A_energies', data=_en_A)
            grp_sys['exc_A_energies'].attrs['description'] = (
                'Q=0 (A) exciton energies; index i gives energy of exciton i+1 (eV). '
                '0.0 for not-loaded slots.')
            grp_sys['exc_A_energies'].attrs['units'] = 'eV'
            grp_sys.create_dataset('exc_B_energies', data=_en_B)
            grp_sys['exc_B_energies'].attrs['description'] = (
                'Q=q (B) exciton energies; index j gives energy of exciton j+1 (eV). '
                'Same as exc_A_energies for Q=0 runs.')
            grp_sys['exc_B_energies'].attrs['units'] = 'eV'

            # ── QP and DFT energy levels ─────────────────────────────────
            grp_en = h5.require_group('energies')
            grp_en.create_dataset('Eqp_cond',  data=Eqp_cond)
            grp_en['Eqp_cond'].attrs['axes']  = '(Nkpoints, Ncbnds)'
            grp_en['Eqp_cond'].attrs['units'] = 'eV'
            grp_en.create_dataset('Eqp_val',   data=Eqp_val)
            grp_en['Eqp_val'].attrs['axes']   = '(Nkpoints, Nvbnds)'
            grp_en['Eqp_val'].attrs['units']  = 'eV'
            grp_en.create_dataset('Edft_cond', data=Edft_cond)
            grp_en['Edft_cond'].attrs['axes'] = '(Nkpoints, Ncbnds)'
            grp_en['Edft_cond'].attrs['units']= 'eV'
            grp_en.create_dataset('Edft_val',  data=Edft_val)
            grp_en['Edft_val'].attrs['axes']  = '(Nkpoints, Nvbnds)'
            grp_en['Edft_val'].attrs['units'] = 'eV'

            # ── run configuration ────────────────────────────────────────
            grp_cfg = h5.require_group('config')
            _skip = {'exciton_pairs', 'dfpt_irreps_list'}  # non-scalar; skip or handle below
            for k, v in config.items():
                if k in _skip:
                    continue
                if isinstance(v, (bool, int, float, str)):
                    grp_cfg.attrs[k] = v
                elif isinstance(v, list):
                    if len(v) > 0:
                        grp_cfg.create_dataset(k, data=np.array(v))
            grp_cfg.attrs['code_version'] = CODE_VERSION

        print(f'Saved forces and metadata to {out_path}')

    ####### Computing forces for each exciton pair — both bases ##########

    forces_ph_RPA      = []
    forces_ph_RPA_diag = []
    forces_ph_Kernel   = []
    forces_ca_RPA      = []
    forces_ca_RPA_diag = []
    forces_ca_Kernel   = []

    time_calc_RPA_diag, time_calc_RPA, time_calc_Kernel = 0.0, 0.0, 0.0

    _unit = -Ry2eV / bohr2A   # Ry/bohr → eV/ang, with sign from F = -dE/dR

    def process_exciton_pair(exc_pair):
        if finite_q_phonon:
            Akcv = Exciton_coeffs_A[excitons_A_to_load.index(exc_pair[0])]
            Bkcv = Exciton_coeffs_B[excitons_B_to_load.index(exc_pair[1])]
        else:
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

    if save_forces_h5:
        time0 = time.clock_gettime(0)
        save_exc_forces_h5(forces_h5_file)
        TASKS.append(['Saving forces to HDF5', time.clock_gettime(0) - time0])

    print_final_msg(TASKS)

