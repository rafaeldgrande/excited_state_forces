
TESTES_DEV = False
verbosity = 'high'

# TODO organize the code!
# TODO save elph_cond and elph_val in h5 format for being reused in a new calc



TASKS = []
TIMING = []

# FIRST MESSAGE

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

    return -f_cart * Ry2eV / bohr2A

def report_forces(iexc, jexc, F_RPA_diag, F_RPA, F_kernel, verbose=False):
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
    arq_out_name = f'forces_cart.out_{iexc_name}_{jexc_name}'
    arq_out = open(arq_out_name, 'w')
        
    text = '''# RPA_diag is equation (1) of arxiv:2502.05144, with d/dr <kcv|K^eh|k'c'v'> = 0
# RPA is equation (3) of arxiv:2502.05144, with  <kcv|d K^eh / dr|k'c'v'> = 0
# RPA_diag_plus_Kernel is equation (1) of arxiv:2502.05144 with  <kcv|d K^eh / dr|k'c'v'> = 0
# Forces units are (eV/ang)\n'''

    arq_out.write(text)
    if verbose:
        print(text)
    header = f'{"Atom":<5} {"dir":<5}     {"RPA_diag":<25}       {"RPA_diag_offdiag":<25}      {"RPA_diag_plus_Kernel":<25}'
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

################################################## code starts here ################################################

if __name__ == "__main__":
    
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
    load_elph_coeffs_hdf5 = config['load_elph_coeffs_hdf5']
    
    if run_parallel == True:
        from multiprocessing import Pool
        from functools import partial
        from itertools import islice

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

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("\n\nExecution date: ", dt_string)
    

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

    read_exciton_pairs(config)

    print("Exciton-ph matrix elements to be computed:")
    for exc_pair in config['exciton_pairs']:
        print(f" <{exc_pair[0]} | dH | {exc_pair[1]}>")
        
    exciton_pairs = config['exciton_pairs']
    # print('exciton_pairs:', exciton_pairs)
    
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

    # getting info from eqp.dat (from absorption calculation)
    time0 = time.clock_gettime(0)
    Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, BSE_params)
    # Shape Eqp_val = (Nkpoints_BSE, Nvbnds)
    # Shape Eqp_cond = (Nkpoints_BSE, Ncbnds)
    time1 = time.clock_gettime(0)
    TASKS.append(['Reading QP and DFT energy levels', time1 - time0])

    # Getting elph coefficients
    # get displacement patterns
    time0 = time.clock_gettime(0)
    iq = 0  # FIXME -> generalize for set of q points. used for excitons with non-zero center of mass momentum
    # Displacements, Nirreps = get_displacement_patterns(el_ph_dir, iq, MF_params)
    Displacements, Nirreps = get_displacement_patterns(el_ph_dir, iq, MF_params)

    # when one wants to use just some irreps, we need to remove the other not used
    if len(dfpt_irreps_list) != 0:
        Displacements = Displacements[dfpt_irreps_list, :, :]
        MF_params.Nmodes = len(dfpt_irreps_list)
        Nmodes = len(dfpt_irreps_list)
        print(f"Not using all irreps. Setting Nmodes to {Nmodes}.")
    else:
        Nmodes = MF_params.Nmodes

    # get elph coefficients from .xml files
    elph, Kpoints_in_elph_file = get_el_ph_coeffs(el_ph_dir, iq, Nirreps, dfpt_irreps_list)
    time1 = time.clock_gettime(0)
    TASKS.append(['Reading ELPH coefficients', time1 - time0])
    
    # imodes_with_no_elph = []
    # for imode in range(Nmodes):
    #     if np.all(elph[imode] == 0):
    #         imodes_with_no_elph.append(imode)
    # if len(imodes_with_no_elph) > 0:
    #     print(f'WARNING! The following modes have no elph coefficients and will be ignored in the forces calculation: {imodes_with_no_elph}')
    #     print('This can happen if you are using just some irreps from the DFPT calculation.')
    #     print('Please, check if this is ok for your calculation.')
    #     print('The number of modes will be reduced accordingly.')
    #     # removing modes with no elph coefficients from Displacements and elph
    #     Displacements = np.delete(Displacements, imodes_with_no_elph, axis=0)
    #     elph = np.delete(elph, imodes_with_no_elph, axis=0)
    #     Nmodes = Nmodes - len(imodes_with_no_elph)
    #     MF_params.Nmodes = Nmodes
    #     print(f'Setting Nmodes to {Nmodes}.')
        
    # print('!!!!!!!!!! Kpoints_in_elph_file !!!!!!!!!!!')
    # print(Kpoints_in_elph_file)
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    Nkpoints_DFPT = len(Kpoints_in_elph_file)

    # change basis for k points from dfpt calculations
    # those k points are in cartesian basis. we're changing 
    # it to reciprocal lattice basis

    time0 = time.clock_gettime(0)
    mat_reclattvecs_to_cart = np.transpose(BSE_params.rec_cell_vecs)
    mat_cart_to_reclattvecs = np.linalg.inv(mat_reclattvecs_to_cart)

    Kpoints_in_elph_file_frac = []

    for ik in range(Nkpoints_DFPT):
        K_cart = mat_cart_to_reclattvecs @ Kpoints_in_elph_file[ik]
        for icomp in range(3):
            K_cart[icomp] = correct_comp_vector(K_cart[icomp])
        Kpoints_in_elph_file_frac.append(K_cart)
        
    Kpoints_in_elph_file_frac = np.array(Kpoints_in_elph_file_frac) # shape (Nkpoints_DFPT, 3)

    if log_k_points == True:
        arq_kpoints_log = open('Kpoints_dfpt_cart_basis', 'w')
        for K_cart in Kpoints_in_elph_file_frac:
            arq_kpoints_log.write(f"{K_cart[0]:.9f}    {K_cart[1]:.9f}     {K_cart[2]:.9f} \n")
        arq_kpoints_log.close()   

    time1 = time.clock_gettime(0)
    TASKS.append(['Changing basis for k points', time1 - time0])


    # filter elph to get just g_c1c2 and g_v1v2, cond and val states that are in the BSE Hamiltonian.
    time0 = time.clock_gettime(0)
    elph_cond, elph_val = filter_elph_coeffs(elph, MF_params, BSE_params) 
    time1 = time.clock_gettime(0)
    TASKS.append(['Filtering ELPH data to ELPH_cond and ELPH_val', time1 - time0])

    # apply acoustic sum rule over elph coefficients
    time0 = time.clock_gettime(0)
    if acoutic_sum_rule == True:
        print('Applying acoustic sum rule for conduction bands. Making sum_mu <i|dH/dmu|j> (mu dot n) = 0 for n = x,y,z.')
        elph_cond = impose_ASR(elph_cond, Displacements, MF_params, acoutic_sum_rule)
    if acoutic_sum_rule == True:
        print('Applying acoustic sum rule for valence bands. Making sum_mu <i|dH/dmu|j> (mu dot n) = 0 for n = x,y,z.')
        elph_val = impose_ASR(elph_val, Displacements, MF_params, acoutic_sum_rule)
    if not acoutic_sum_rule:
        print('Not applying acoustic sum rule over elph coefficients. Check later if the force on the center of mass is negligible!')
    time1 = time.clock_gettime(0)
    TASKS.append(['Impose ASR on ELPH coefficients.', time1 - time0])

    # Let's put all k points from BSE grid in the first Brillouin zone
    ikBSE_to_ikDFPT = translate_bse_to_dfpt_k_points()

    del elph
    report_ram()


    # renormalze elph coefficients
    time0 = time.clock_gettime(0)
    if no_renorm_elph == False:
        print('Renormalizing ELPH coefficients following equation (5) of arxiv:2502.05144') 
        print('where <n|dHqp|m> = <n|dHdft|m>(Eqp_n - Eqp_m)/(Edft_n - Edft_m) when Edft_n != Edft_m')
        print('and <n|dHqp|m> = <n|dHdft|m> otherwise')
        elph_cond, elph_val = renormalize_elph_considering_kpt_order(elph_cond, elph_val, Eqp_val, Eqp_cond, Edft_val, 
                                                                     Edft_cond, MF_params, BSE_params, ikBSE_to_ikDFPT)
    else:
        print('Not Renormalizing ELPH coefficients. Using <n|dHqp|m> = <n|dHdft|m> for all n and m')


    time1 = time.clock_gettime(0)
    TASKS.append(['Renormalization of ELPH coefficients', time1 - time0])
    print('\n\n')


    # interpolation of elph coefficients a la BerkeleyGW code?
    time0 = time.clock_gettime(0)
    if elph_fine_a_la_bgw == False:
        
        print('No interpolation on elph coeffs from a coarse grid to a fine grid is used')
        
        if trust_kpoints_order == False:
            print('Checking if kpoints of DFPT and BSE agree with each other')

            # Checking kpoints from DFPT and BSE calculations
            # The kpoints in eigenvecs.h5 are not in the same order in the
            # input for the fine grid calculation.
            # The k points in BSE are reported in reciprocal lattice vectors basis
            # and in DFPT those k points are reported in cartersian basis in units
            # of reciprocal lattice

            # It SEEMS that the order of k points in the eqp.dat (produced by the absorption code)
            # is the same than the order of k points in the eigenvecs file
            # Maybe it would be necessary to check it later!

            # Now checking if everything is ok with ikBSE_to_ikDFPT list
            # if something is wrong kill the code
            check_k_points_BSE_DFPT()
        else:
            print('As trust_kpoints_order is true, I am not mapping k points from BSE with k points from DFPT.')
            print('I am assuming they are informed in the same order in both calculations')
        
    else:
        
        # comment that one need to use a modified version of the BGW code to get those files
        print('Using interpolation "a la BerkeleyGW code"')
        print('Reading coefficients relating fine and coarse grids from files dtmat_non_bin_val and dtmat_non_bin_conds')

        elph_cond = elph_interpolate_bgw(elph_cond, 'dtmat_non_bin_cond', BSE_params.Nkpoints_BSE, BSE_params.Ncbnds)
        elph_val  = elph_interpolate_bgw(elph_val, 'dtmat_non_bin_val', BSE_params.Nkpoints_BSE, BSE_params.Nvbnds)

    time1 = time.clock_gettime(0)
    TASKS.append(['Interpolation of ELPH coefficients', time1 - time0])
    
    # save elph coefficients in hdf5 file for being reused in a new calculation
    if save_elph_coeffs == True:
        time0 = time.clock_gettime(0)
        print('\nSaving elph coefficients in hdf5 files\n')
        save_elph_coeffs_hdf5(elph_cond, elph_val, elph_fine_a_la_bgw, no_renorm_elph, 'elph_coeffs.h5')
        time1 = time.clock_gettime(0)
        TASKS.append(['Saving ELPH coefficients in hdf5 file', time1 - time0])


    time0 = time.clock_gettime(0)
    print("\nExpanding ELPH matrices for vectorized multiplication")

    Shape_augmented = (Nmodes, Nkpoints_BSE, Ncbnds, Nvbnds, Ncbnds, Nvbnds)
    Gc = np.zeros(Shape_augmented, dtype=np.complex64)
    Gv = np.zeros(Shape_augmented, dtype=np.complex64)

    Shape_augmented_diag = (Nmodes, Nkpoints_BSE, Ncbnds, Nvbnds)
    Gc_diag = np.zeros(Shape_augmented_diag, dtype=np.complex64)
    Gv_diag = np.zeros(Shape_augmented_diag, dtype=np.complex64)

    print(f"Old shape for cond elph is {elph_cond.shape}. New shape is {Gc.shape} including off-diagonal elements and {Gc_diag.shape} for diagonal elements")
    print(f"Old shape for val elph is {elph_val.shape}. New shape is {Gv.shape} including off-diagonal elements and {Gv_diag.shape} for diagonal elements")

    # For Gc: Gc[imode, ik, ic1, iv, ic2, iv] = elph_cond[imode, ik, ic1, ic2]
    # Only nonzero when iv == iv, so fill all iv
    for iv in range(Nvbnds):
        Gc[:, :, :, iv, :, iv] = elph_cond

    # For Gv: Gv[imode, ik, ic, iv1, ic, iv2] = elph_val[imode, ik, iv1, iv2]
    # Only nonzero when ic == ic, so fill all ic
    for ic in range(Ncbnds):
        Gv[:, :, ic, :, ic, :] = elph_val
        
    # apply Q shift on valence states. Used when having finite momentum excitons.
    Gv = apply_Qshift_on_valence_states(Qshift, Gv, Kpoints_in_elph_file_frac)

    # Vectorized assignment for Gc_diag and Gv_diag
    # Gc_diag[imode, ik, ic, iv] = elph_cond[imode, ik, ic, ic] for all imode, ik, ic, iv
    # Gv_diag[imode, ik, ic, iv] = elph_val[imode, ik, iv, iv] for all imode, ik, ic, iv

    # For Gc_diag: broadcast elph_cond diagonal over iv
    diag_elph_cond = np.diagonal(elph_cond, axis1=2, axis2=3)  # shape: (Nmodes, Nkpoints_BSE, Ncbnds)
    Gc_diag[:] = diag_elph_cond[:, :, :, np.newaxis]  # broadcast over iv. shape = (Nmodes, Nkpoints_BSE, Ncbnds, Nvbnds)

    # For Gv_diag: broadcast elph_val diagonal over ic
    diag_elph_val = np.diagonal(elph_val, axis1=2, axis2=3)  # shape: (Nmodes, Nkpoints_BSE, Nvbnds)
    Gv_diag[:] = diag_elph_val[:, :, np.newaxis, :]  # broadcast over ic. shape = (Nmodes, Nkpoints_BSE, Ncbnds, Nvbnds)

    time1 = time.clock_gettime(0)
    TASKS.append(['ELPH matrices expansion (for vectorized multiplication)', time1 - time0])
    
    dRPA_dr_imode_mat = Gc - Gv
    dRPA_dr_imode_diag_mat = Gc_diag - Gv_diag
    
    
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
        DKernel_dr_imode_mat = calc_Dkernel_new(kernel_matrix, elph_cond, elph_val, Eqp_cond, Eqp_val, vectorized=do_vectorized_sums) # units ry/bohr
        time1 = time.clock_gettime(0)
        TASKS.append(['Calculating d/dr_imode <kcv|K|kc\'v\'> matrix elements', time1 - time0])

    ####### Computing forces for each exciton pair and each mode ##########
    
    forces_A_B_RPA = []
    forces_A_B_RPA_diag = []
    forces_A_B_Kernel = []
    
    time_calc_RPA_diag, time_calc_RPA, time_calc_Kernel = 0.0, 0.0, 0.0

    # Function to process a single exciton pair
    def process_exciton_pair(exc_pair):
        Akcv, Bkcv = Exciton_coeffs[exc_pair[0]-1], Exciton_coeffs[exc_pair[1]-1]
        
        time0 = time.clock_gettime(0)
        forces_A_B = compute_A_dRPA_dr_imode_B(Akcv, Bkcv, dRPA_dr_imode_mat, elph_cond, elph_val, vectorized=do_vectorized_sums)
        rpa = f_disp_to_cart_basis(forces_A_B, Displacements)
        time_rpa = time.clock_gettime(0) - time0
        
        time0 = time.clock_gettime(0)
        forces_A_B = compute_A_dRPAdiag_dr_imode_B(Akcv, Bkcv, dRPA_dr_imode_diag_mat, elph_cond, elph_val, vectorized=do_vectorized_sums)
        rpa_diag = f_disp_to_cart_basis(forces_A_B, Displacements)
        time_rpa_diag = time.clock_gettime(0) - time0
        
        kernel = None
        time_kernel = 0.0
        if Calculate_Kernel == True:
            time0 = time.clock_gettime(0)
            forces_A_B = compute_A_dKernel_dr_imode_B(Akcv, Bkcv, DKernel_dr_imode_mat, vectorized=do_vectorized_sums)
            kernel = f_disp_to_cart_basis(forces_A_B, Displacements)
            time_kernel = time.clock_gettime(0) - time0
        
        return exc_pair, rpa, rpa_diag, kernel, time_rpa, time_rpa_diag, time_kernel
    
    total_pairs = len(exciton_pairs)
    # running in parallel or serial
    if run_parallel == False:
        print("\n\n################################# Running in serial ################################")
        print("Total exciton-phonon matrix elements to be calculated: ", total_pairs)
        for i_pair, exc_pair in enumerate(exciton_pairs):
            if i_pair > 0 and (i_pair == 1 or i_pair == 5 or i_pair % 10 == 0 or i_pair == total_pairs - 1):
                progress = i_pair / total_pairs * 100
                print(f"Progress excited-state forces calculation: {progress:.2f}% ({i_pair}/{total_pairs})")
            exc_pair, rpa, rpa_diag, kernel, time_rpa, time_rpa_diag, time_kernel = process_exciton_pair(exc_pair)
            forces_A_B_RPA.append(rpa)
            forces_A_B_RPA_diag.append(rpa_diag)
            if kernel is not None:
                forces_A_B_Kernel.append(kernel)
            time_calc_RPA += time_rpa
            time_calc_RPA_diag += time_rpa_diag
            time_calc_Kernel += time_kernel
                
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
                    progress = i_pair / total_pairs * 100
                    print(f"Progress excited-state forces calculation: {progress:.2f}% ({i_pair}/{total_pairs})")
        
        for exc_pair, rpa, rpa_diag, kernel, time_rpa, time_rpa_diag, time_kernel in results:
            forces_A_B_RPA.append(rpa)
            forces_A_B_RPA_diag.append(rpa_diag)
            if kernel is not None:
                forces_A_B_Kernel.append(kernel)
            time_calc_RPA += time_rpa
            time_calc_RPA_diag += time_rpa_diag
            time_calc_Kernel += time_kernel
            
    TASKS.append(['Calculating forces with RPA_diag', time_calc_RPA_diag])
    TASKS.append(['Calculating forces with RPA', time_calc_RPA])
    TASKS.append(['Calculating forces with Kernel part', time_calc_Kernel])
            
    forces_A_B_RPA_diag = np.array(forces_A_B_RPA_diag)
    forces_A_B_RPA = np.array(forces_A_B_RPA)
    if Calculate_Kernel == True:
        forces_A_B_Kernel = np.array(forces_A_B_Kernel)
        
    for i, exc_pair in enumerate(exciton_pairs):
        iexc, jexc = exc_pair
        report_forces(iexc-1, jexc-1, forces_A_B_RPA_diag[i], forces_A_B_RPA[i], forces_A_B_Kernel[i] if Calculate_Kernel else None, verbose=len(exciton_pairs)==1)
        
    

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
