
TESTES_DEV = False
verbosity = 'high'

# TODO oranize the code!
# TODO format prints from forces calculations!
# TODO in the beging say how many calculation wil done and how much I would have done if used every thing,


TASKS = []
TIMING = []

run_parallel = False
if run_parallel == True:
    from multiprocessing import Pool
    from multiprocessing import freeze_support

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

# datetime object containing current date and time
now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("\n\nExecution date: ", dt_string)

print('\n\n*************************************************************')
print('Excited state forces code')
print('Developed by Rafael Del Grande and David Strubbe')
print('*************************************************************\n\n')


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
            

            found_or_not = find_kpoint(vec_eigvecs, Kpoints_in_elph_file_cart)
            # if found the vec_eigvecs in the Kpoints_in_elph_file_cart, then returns
            # the index in the Kpoints_in_elph_file_cart.
            # if did not find it, then returns -1

            # the conversion list from one to another
            ikBSE_to_ikDFPT.append(found_or_not)

        # debug
        arq_teste.close()
        
    else:
        for ik in range(len(Kpoints_in_elph_file_cart)):
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


######################### RUNNING CODE ##############################

start_time = time.clock_gettime(0)
# Getting BSE and MF parameters
# Reading eigenvecs.h5 file
time0 = time.clock_gettime(0)
Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs, BSE_params, MF_params = get_BSE_MF_params()
time1 = time.clock_gettime(0)
TASKS.append(['Get parameters from DFT and GWBSE', time1 - time0])


# getting info from eqp.dat (from absorption calculation)
time0 = time.clock_gettime(0)
Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, BSE_params)
time1 = time.clock_gettime(0)
TASKS.append(['Reading QP and DFT energy levels', time1 - time0])

# Getting elph coefficients
# get displacement patterns
time0 = time.clock_gettime(0)
iq = 0  # FIXME -> generalize for set of q points. used for excitons with non-zero center of mass momentum
Displacements, Nirreps = get_displacement_patterns(iq, MF_params)

# when one wants to use just some irreps, we need to remove the other not used
if len(dfpt_irreps_list) != 0:
    Displacements = Displacements[dfpt_irreps_list, :, :]
    MF_params.Nmodes = len(dfpt_irreps_list)
    Nmodes = len(dfpt_irreps_list)
    print(f"Not using all irreps. Setting Nmodes to {Nmodes}.")
else:
    Nmodes = MF_params.Nmodes

# get elph coefficients from .xml files

elph, Kpoints_in_elph_file = get_el_ph_coeffs(iq, Nirreps, dfpt_irreps_list)
time1 = time.clock_gettime(0)
TASKS.append(['Reading ELPH coefficients', time1 - time0])

Nkpoints_DFPT = len(Kpoints_in_elph_file)

# change basis for k points from dfpt calculations
# those k points are in cartesian basis. we're changing 
# it to reciprocal lattice basis

time0 = time.clock_gettime(0)
mat_reclattvecs_to_cart = np.transpose(BSE_params.rec_cell_vecs)
mat_cart_to_reclattvecs = np.linalg.inv(mat_reclattvecs_to_cart)

Kpoints_in_elph_file_cart = []

for ik in range(Nkpoints_DFPT):
    K_cart = mat_cart_to_reclattvecs @ Kpoints_in_elph_file[ik]
    for icomp in range(3):
        K_cart[icomp] = correct_comp_vector(K_cart[icomp])
    Kpoints_in_elph_file_cart.append(K_cart)
    
Kpoints_in_elph_file_cart = np.array(Kpoints_in_elph_file_cart)

if log_k_points == True:
    arq_kpoints_log = open('Kpoints_dfpt_cart_basis', 'w')
    for K_cart in Kpoints_in_elph_file_cart:
        arq_kpoints_log.write(f"{K_cart[0]:.9f}    {K_cart[1]:.9f}     {K_cart[2]:.9f} \n")
    arq_kpoints_log.close()   

time1 = time.clock_gettime(0)
TASKS.append(['Changing basis for k points', time1 - time0])


# filter data to get just g_c1c2 and g_v1v2
time0 = time.clock_gettime(0)
elph_cond, elph_val = filter_elph_coeffs(elph, MF_params, BSE_params) 
time1 = time.clock_gettime(0)
TASKS.append(['Filtering ELPH data to ELPH_cond and ELPH_val', time1 - time0])

# apply acoustic sum rule over elph coefficients
time0 = time.clock_gettime(0)
elph_cond = impose_ASR(elph_cond, Displacements, MF_params, acoutic_sum_rule)
elph_val = impose_ASR(elph_val, Displacements, MF_params, acoutic_sum_rule)
time1 = time.clock_gettime(0)
TASKS.append(['Impose ASR on ELPH coefficients', time1 - time0])

# Let's put all k points from BSE grid in the first Brillouin zone
ikBSE_to_ikDFPT = translate_bse_to_dfpt_k_points()

del elph
report_ram()


# renormalze elph coefficients
time0 = time.clock_gettime(0)
if no_renorm_elph == False:
    print('Renormalizing ELPH coefficients') 
    print('where <n|dHqp|m> = <n|dHdft|m>(Eqp_n - Eqp_m)/(Edft_n - Edft_m) when Edft_n != Edft_m')
    print('and <n|dHqp|m> = <n|dHdft|m> otherwise')
else:
    print('Not Renormalizing ELPH coefficients. Using <n|dHqp|m> = <n|dHdft|m> for all n and m')
elph_cond, elph_val = renormalize_elph_considering_kpt_order(elph_cond, elph_val, Eqp_val, Eqp_cond, Edft_val, Edft_cond, 
                                                                           MF_params, BSE_params, ikBSE_to_ikDFPT)

time1 = time.clock_gettime(0)
TASKS.append(['Renormalization of ELPH coefficients', time1 - time0])

# interpolation of elph coefficients a la BerkeleyGW code?
time0 = time.clock_gettime(0)
if elph_fine_a_la_bgw == False:
    
    print('No interpolation on elph coeffs is used')
    
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
    
    print('Using interpolation "a la BerkeleyGW code"')
    print('Reading coefficients relating fine and coarse grids from files dtmat_non_bin_val and dtmat_non_bin_conds')

    elph_cond = elph_interpolate_bgw(elph_cond, 'dtmat_non_bin_cond', BSE_params.Nkpoints_BSE, BSE_params.Ncbnds)
    elph_val  = elph_interpolate_bgw(elph_val, 'dtmat_non_bin_val', BSE_params.Nkpoints_BSE, BSE_params.Nvbnds)

time1 = time.clock_gettime(0)
TASKS.append(['Interpolation of ELPH coefficients', time1 - time0])

time0 = time.clock_gettime(0)

print("Expanding ELPH matrices for vectorized multiplication")

Shape_augmented = (Nmodes, Nkpoints_BSE, Ncbnds, Nvbnds, Ncbnds, Nvbnds)
Gc = np.zeros(Shape_augmented, dtype=np.complex64)
Gv = np.zeros(Shape_augmented, dtype=np.complex64)

Shape_augmented_diag = (Nmodes, Nkpoints_BSE, Ncbnds, Nvbnds)
Gc_diag = np.zeros(Shape_augmented_diag, dtype=np.complex64)
Gv_diag = np.zeros(Shape_augmented_diag, dtype=np.complex64)

print(f"Old shape for cond elph is {elph_cond.shape}. New shape is {Gc.shape} including off-diagonal elements and {Gc_diag.shape} for diagonal elements")
print(f"Old shape for val elph is {elph_val.shape}. New shape is {Gv.shape} including off-diagonal elements and {Gv_diag.shape} for diagonal elements")

for imode in range(Nmodes):
    for ik in range(Nkpoints_BSE):
        
        for ic in range(Ncbnds):
            for iv in range(Nvbnds):        
                Gc_diag[imode, ik, ic, iv] = elph_cond[imode, ik, ic, ic]
                Gv_diag[imode, ik, ic, iv] = elph_val[imode, ik, iv, iv]
        
        for iv in range(Nvbnds):
            for ic1 in range(Ncbnds):
                for ic2 in range(Ncbnds):
                    Gc[imode, ik, ic1, iv, ic2, iv] = elph_cond[imode, ik, ic1, ic2]
        
        for ic in range(Ncbnds):
            for iv1 in range(Nvbnds):
                for iv2 in range(Nvbnds):
                    Gv[imode, ik, ic, iv1, ic, iv2] = elph_val[imode, ik, iv1, iv2]

time1 = time.clock_gettime(0)
TASKS.append(['ELPH matrices expansion (for vectorized multiplication)', time1 - time0])

# Loading exciton coefficients

# Getting exciton info

def load_exciton_coeffs(iexc, jexc):
    time0 = time.clock_gettime(0)
    Akcv, OmegaA, Bkcv, OmegaB = get_exciton_coeffs(iexc, jexc)
    report_expected_energies_master(iexc, jexc, Eqp_cond, Eqp_val, Akcv, OmegaA, Bkcv, OmegaB)
        
    # limited sums of BSE coefficients
    indexes_limited_BSE_sum = generate_indexes_limited_BSE_sum()

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
    summarize_Acvk(Akcv, BSE_params.Kpoints_BSE, indexes_limited_BSE_sum)

    delta_time = time.clock_gettime(0) - time0
    return Akcv, OmegaA, Bkcv, OmegaB, indexes_limited_BSE_sum, delta_time
    
TASKS.append(['Loading exciton coefficients', time1 - time0])
   
### Loading Kernel matrix elements
# Getting kernel info from bsemat.h5 file
if Calculate_Kernel == True:
    time0 = time.clock_gettime(0)
    Kd, Kx = get_kernel(kernel_file, factor_head)
    time1 = time.clock_gettime(0)
    TASKS.append(['Loading kernel matrix elements', time1 - time0])


########## Calculating stuff ############

print("\n\nCalculating matrix elements for forces calculations <cvk|dH/dx_mu|c'v'k'>")

def calculate_excited_state_forces(Akcv, Bkcv, indexes_limited_BSE_sum):
    
    time0 = time.clock_gettime(0)

    print(f"     Calculating exciton coefficient {iexc} {jexc}")
    if do_vectorized_sums == True:
        
        
        
        Sum_DKinect_diag    = np.zeros((Nmodes), dtype=np.complex64)
        Sum_DKinect_offdiag = np.zeros((Nmodes), dtype=np.complex64)
        Sum_DKernel         = np.zeros((Nmodes), dtype=np.complex64)
        
        # build A_mat[ik, ic1, iv1, ic2, iv2] = Akcv[ik, ic1, iv1] * np.conj(Bkcv[ik, ic2, iv2])
        A_mat = Akcv[:, :, :, np.newaxis, np.newaxis] * np.conj(Bkcv[:, np.newaxis, np.newaxis, :, :])
        A_mat_diag = Akcv * np.conj(Bkcv)
        
        print('Using vectorized sums for forces calculations')
        print('Creating matrix A_mat[ik, ic1, iv1, ic2, iv2] = Akcv[ik, ic1, iv1] * np.conj(Bkcv[ik, ic2, iv2])')
        print('Creating matrix A_mat_diag[ik, ic, iv] = Akcv[ik, ic, iv] * np.conj(Bkcv[ik, ic, iv]) for diagonal approximation')
        
        for imode in range(Nmodes):
            
            # Multiply A_mat * (Gc[imode] - Gv[imode])
            
            F_imode = A_mat * (Gc[imode] - Gv[imode])
            
            # F_imode_diag = np.diagonal(F_imode, axis1=1, axis2=2)
            # F_imode_diag = np.diagonal(F_imode_diag, axis1=2, axis2=3)  # Now shape (nk, nc, nv)
            F_imode_diag = A_mat_diag * (Gc_diag[imode] - Gv_diag[imode])

            Sum_DKinect_diag[imode] = np.sum(F_imode_diag)
            Sum_DKinect_offdiag[imode] = np.sum(F_imode) - Sum_DKinect_diag[imode]

    else:
        
        print('Not using vectorized sums for forces calculations')
        # instead of creating big matrix, calculate sums on the fly!

        print('Creating list of indexes kcv for which sums are calculated')
        args_list_just_diag, args_list_just_offdiag = arg_lists_Dkinect(BSE_params, indexes_limited_BSE_sum)


        print("")
        Sum_DKinect_diag    = np.zeros((Nmodes), dtype=np.complex64)
        Sum_DKinect_offdiag = np.zeros((Nmodes), dtype=np.complex64)
        Sum_DKernel         = np.zeros((Nmodes), dtype=np.complex64)

        print('\n\nCalculating diagonal matrix elements <kcv|dH/dx_mu|kcv>')
        for imode in range(Nmodes):
            print(f"Calculating mode {imode + 1} of {Nmodes}")
            Sum_DKinect_diag[imode] = calc_Dkinect_matrix_simplified(Akcv, Bkcv, elph_cond, elph_val, args_list_just_diag, imode)
        
        print("\n\nCalculating off-diagonal matrix elements <kcv|dH/dx_mu|kc'v'>") 
        for imode in range(Nmodes):
            print(f"Calculating mode {imode + 1} of {Nmodes}")
            Sum_DKinect_offdiag[imode] = calc_Dkinect_matrix_simplified(Akcv, Bkcv, elph_cond, elph_val, args_list_just_offdiag, imode)

        # Kernel derivatives
        if Calculate_Kernel == True:

            EDFT = Edft_val, Edft_cond
            EQP = Eqp_val, Eqp_cond
            ELPH = elph_cond, elph_val

            DKernel = calc_deriv_Kernel((Kx+Kd)*Ry2eV, EDFT, EQP, ELPH, Akcv, Bkcv, MF_params, BSE_params)

            for imode in range(Nmodes):
                Sum_DKernel[imode] = np.sum(DKernel[imode])

        report_ram()
    
    delta_time = time.clock_gettime(0) - time0
    
    return Sum_DKinect_diag, Sum_DKinect_offdiag, Sum_DKernel, delta_time

def report_forces(iexc, jexc, Sum_DKinect_diag, Sum_DKinect_offdiag, Sum_DKernel):
    # Convert from Ry/bohr to eV/A. Minus sign comes from F=-dV/du

    Sum_DKinect_diag    = -Sum_DKinect_diag * Ry2eV / bohr2A
    Sum_DKinect_offdiag = -Sum_DKinect_offdiag * Ry2eV / bohr2A
    Sum_DKernel = -Sum_DKernel * Ry2eV / bohr2A

    # Warn if imag part is too big (>= 10^-6)

    if max(abs(np.imag(Sum_DKinect_diag))) >= 1e-6:
        print('WARNING: Imaginary part of kinectic diagonal forces >= 10^-6 eV/angs!')

    if max(abs(np.imag(Sum_DKinect_offdiag))) >= 1e-6:
        print('WARNING: Imaginary part of kinectic offdiagonal forces >= 10^-6 eV/angs!')

    if Calculate_Kernel == True:
        if max(abs(np.imag(Sum_DKernel))) >= 1e-6:
            print('WARNING: Imaginary part of Kernel forces >= 10^-6 eV/angs!')

    # Show just real part of numbers (default)    
    if iexc == jexc:                                                                                                                                                                                                                                                                                                                    
        if show_imag_part == False:
            Sum_DKinect_diag = np.real(Sum_DKinect_diag)
            Sum_DKinect_offdiag = np.real(Sum_DKinect_offdiag)
            Sum_DKernel = np.real(Sum_DKernel)

    # Calculate forces cartesian basis

    print("Calculating forces in cartesian basis")

    F_cart_KE_IBL = np.zeros((Nat, 3), dtype=complex)  # IBL just diag RPA
    # david thesis = diag + offdiag from kinect part
    F_cart_KE_David = np.zeros((Nat, 3), dtype=complex)
    F_cart_Kernel_IBL = np.zeros((Nat, 3), dtype=complex)

    for iatom in range(Nat):
        for imode in range(Nmodes):
            F_cart_KE_IBL[iatom]   = F_cart_KE_IBL[iatom] + Displacements[imode,iatom] * Sum_DKinect_diag[imode]
            F_cart_KE_David[iatom] = F_cart_KE_David[iatom] + Displacements[imode,iatom] * (Sum_DKinect_offdiag[imode] + Sum_DKinect_diag[imode])
            F_cart_Kernel_IBL[iatom] = F_cart_Kernel_IBL[iatom] + Displacements[imode,iatom] * (Sum_DKernel[imode] + Sum_DKinect_diag[imode])
            # need to make x = x + y, instead of x += y because numpy complains that x+=y does not work when type(x)=!type(y) (one is complex and the other is real - float)

    # Reporting forces in cartesian basis
    DIRECTION = ['x', 'y', 'z']

    arq_out = open(f'forces_cart.out_{iexc}_{jexc}', 'w')

    print('\n\nForces (eV/ang)\n')

    if Calculate_Kernel:
        header = '# Atom  dir  RPA_diag        RPA_diag_offiag    RPA_diag_Kernel'
        print(header)
        arq_out.write(header + '\n')
    else:
        header = '# Atom  dir  RPA_diag        RPA_diag_offiag'
        print(header)
        arq_out.write(header + '\n')

    for iatom in range(Nat):
        for idir in range(3):
            text = f'{iatom+1:<5} {DIRECTION[idir]:<5}     {F_cart_KE_IBL[iatom, idir]:<20.8f}       {F_cart_KE_David[iatom, idir]:<20.8f}'
            if Calculate_Kernel:
                text += f' {F_cart_Kernel_IBL[iatom, idir]:<20.8f}'
            print(text)
            arq_out.write(text + '\n')

    arq_out.close()

report_ram()
# stopping the library
tracemalloc.stop()

delta_time_load_excitons_coeffs = 0.0
delta_time_calculate_excited_state_forces = 0.0

for iexc, jexc in exciton_pairs:
    Akcv, OmegaA, Bkcv, OmegaB, indexes_limited_BSE_sum, delta_time = load_exciton_coeffs(iexc, jexc)
    delta_time_load_excitons_coeffs += delta_time
    Sum_DKinect_diag, Sum_DKinect_offdiag, Sum_DKernel, delta_time = calculate_excited_state_forces(Akcv, Bkcv, indexes_limited_BSE_sum)
    delta_time_calculate_excited_state_forces += delta_time
    report_forces(iexc, jexc, Sum_DKinect_diag, Sum_DKinect_offdiag, Sum_DKernel)
    
TASKS.append(['Loading exciton coefficients', delta_time_load_excitons_coeffs])
TASKS.append(['Calculation of forces', delta_time_calculate_excited_state_forces])


print("\n\n\n\n")
TOTAL_TIME = 0.0

# Set the column widths
task_column_width = 50
time_column_width = 10

print(f"{'Task':<{task_column_width}}{'Time (seconds)':>{time_column_width}}")  # Header
print("-" * (task_column_width + time_column_width))  # Divider

for task in TASKS:
    TOTAL_TIME += task[1]
    print(f"{task[0]:<{task_column_width}}{task[1]:>{time_column_width}.2f}")

print("-" * (task_column_width + time_column_width))  # Divider
print(f"{'Total':<{task_column_width}}{TOTAL_TIME:>{time_column_width}.2f}")


