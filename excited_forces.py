
TESTES_DEV = False
verbosity = 'high'

# TODO oranize the code!
# TODO format prints from forces calculations!
# TODO in the beging say how many calculation wil done and how much I would have done if used every thing,

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
Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs, BSE_params, MF_params = get_BSE_MF_params()

# getting info from eqp.dat (from absorption calculation)
Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, BSE_params)


# Getting elph coefficients
# get displacement patterns
iq = 0  # FIXME -> generalize for set of q points. used for excitons with non-zero center of mass momentum
Displacements, Nirreps = get_displacement_patterns(iq, MF_params)

# when one wants to use just some irreps, we need to remove the other not used
if len(dfpt_irreps_list) != 0:
    Displacements = Displacements[dfpt_irreps_list, :, :]
    MF_params.Nmodes = len(dfpt_irreps_list)
    Nmodes = len(dfpt_irreps_list)
    print(f"Not using all irreps. Setting Nmodes to {Nmodes}.")

# get elph coefficients from .xml files
# if run_parallel == False:

elph, Kpoints_in_elph_file = get_el_ph_coeffs(iq, Nirreps, dfpt_irreps_list)
# else:
#     elph, Kpoints_in_elph_file = get_el_ph_coeffs_parallel(iq, Nirreps)
Nkpoints_DFPT = len(Kpoints_in_elph_file)

# change basis for k points from dfpt calculations
# those k points are in cartesian basis. we're changing 
# it to reciprocal lattice basis

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


# filter data to get just g_c1c2 and g_v1v2
elph_cond, elph_val = filter_elph_coeffs(elph, MF_params, BSE_params) 

# apply acoustic sum rule over elph coefficients
elph_cond = impose_ASR(elph_cond, Displacements, MF_params, acoutic_sum_rule)
elph_val = impose_ASR(elph_val, Displacements, MF_params, acoutic_sum_rule)

# Let's put all k points from BSE grid in the first Brillouin zone
ikBSE_to_ikDFPT = translate_bse_to_dfpt_k_points()

del elph
report_ram()

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

# Loading exciton coefficients

# Getting exciton info
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
   
### Loading Kernel matrix elements
# Getting kernel info from bsemat.h5 file
if Calculate_Kernel == True:
    Kd, Kx = get_kernel(kernel_file, factor_head)


########## Calculating stuff ############

print("\n\nCalculating matrix elements for forces calculations <cvk|dH/dx_mu|c'v'k'>")

# creating KCV list with indexes ik, ic, iv (in this order) used to vectorize future sums
# KCV_list = []

# for ik in range(BSE_params.Nkpoints_BSE):
#     for ic in range(BSE_params.Ncbnds_sum):
#         for iv in range(BSE_params.Nvbnds_sum):
#             KCV_list.append((ik, ic, iv))            


# Creating auxialiry quantities
# aux_cond_matrix[imode, ik, ic1, ic2] = elph_cond[imode, ik, ic1, ic2] * deltaEqp / deltaEdft (if ic1 != ic2)
# aux_val_matrix[imode, ik, iv1, iv2]  = elph_val[imode, ik, iv1, iv2]  * deltaEqp / deltaEdft (if iv1 != iv2)
# If ic1 == ic2 (iv1 == iv2), then the matrix elements are just the elph coefficients"""
if no_renorm_elph == False:
    print('Renormalizing ELPH coefficients') 
    print('where <n|dHqp|m> = <n|dHdft|m>(Eqp_n - Eqp_m)/(Edft_n - Edft_m) when Edft_n != Edft_m')
    print('and <n|dHqp|m> = <n|dHdft|m> otherwise')
else:
    print('Not Renormalizing ELPH coefficients. Using <n|dHqp|m> = <n|dHdft|m> for all n and m')

aux_cond_matrix, aux_val_matrix = aux_matrix_elem(
    elph_cond, elph_val, Eqp_val, Eqp_cond, Edft_val, Edft_cond, MF_params, BSE_params, ikBSE_to_ikDFPT)

# Calculating matrix elements F_cvkc'v'k'
# DKinect = calc_Dkinect_matrix(
#     Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, MF_params, BSE_params)

# instead of creating big matrix, calculate sums on the fly!

print('Creating list of indexes kcv for which sums are calculated')
args_list_just_diag, args_list_just_offdiag = arg_lists_Dkinect(BSE_params, indexes_limited_BSE_sum)


print("")
Sum_DKinect_diag, Sum_DKinect_offdiag = [], []

print('\n\nCalculating diagonal matrix elements <kcv|dH/dx_mu|kcv>')
for imode in range(Nmodes):
    print(f"Calculating mode {imode + 1} of {Nmodes}")
    Sum_DKinect_diag.append(calc_Dkinect_matrix_simplified(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, args_list_just_diag, imode))
   
print("\n\nCalculating off-diagonal matrix elements <kcv|dH/dx_mu|kc'v'>") 
for imode in range(Nmodes):
    print(f"Calculating mode {imode + 1} of {Nmodes}")
    Sum_DKinect_offdiag.append(calc_Dkinect_matrix_simplified(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, args_list_just_offdiag, imode))        

Sum_DKinect_diag, Sum_DKinect_offdiag = np.array(Sum_DKinect_diag), np.array(Sum_DKinect_offdiag)

# dont need aux_cond_matrix, aux_val_matrix anymores
del aux_cond_matrix, aux_val_matrix

# Kernel derivatives
if Calculate_Kernel == True:

    EDFT = Edft_val, Edft_cond
    EQP = Eqp_val, Eqp_cond
    ELPH = elph_cond, elph_val

    DKernel = calc_deriv_Kernel((Kx+Kd)*Ry2eV, EDFT, EQP, ELPH, Akcv, Bkcv, MF_params, BSE_params)

    # dont need the kernel matrix anymore
    del Kx, Kd
    
    Sum_DKernel = np.zeros((Nmodes), dtype=np.complex64)

    for imode in range(Nmodes):
        Sum_DKernel[imode] = np.sum(DKernel[imode])

report_ram()

# Convert from Ry/bohr to eV/A. Minus sign comes from F=-dV/du

Sum_DKinect_diag    = -Sum_DKinect_diag * Ry2eV / bohr2A
Sum_DKinect_offdiag = -Sum_DKinect_offdiag * Ry2eV / bohr2A

if Calculate_Kernel == True:
    Sum_DKernel = -Sum_DKernel*Ry2eV/bohr2A

# Warn if imag part is too big (>= 10^-6)

if max(abs(np.imag(Sum_DKinect_diag))) >= 1e-6:
    print('WARNING: Imaginary part of kinectic diagonal forces >= 10^-6 eV/angs!')

if max(abs(np.imag(Sum_DKinect_offdiag))) >= 1e-6:
    print('WARNING: Imaginary part of kinectic offdiagonal forces >= 10^-6 eV/angs!')

if Calculate_Kernel == True:
    if max(abs(np.imag(Sum_DKernel))) >= 1e-6:
        print('WARNING: Imaginary part of Kernel forces >= 10^-6 eV/angs!')

# Show just real part of numbers (default)                                                                                                                                                                                                                                                                                                                        
if show_imag_part == False:
    Sum_DKinect_diag = np.real(Sum_DKinect_diag)
    Sum_DKinect_offdiag = np.real(Sum_DKinect_offdiag)

    if Calculate_Kernel == True:
        Sum_DKernel = np.real(Sum_DKernel)

# Calculate forces cartesian basis

print("Calculating forces in cartesian basis")

if show_imag_part == True:
    F_cart_KE_IBL = np.zeros((Nat, 3), dtype=complex)  # IBL just diag RPA
    # david thesis = diag + offdiag from kinect part
    F_cart_KE_David = np.zeros((Nat, 3), dtype=complex)
    if Calculate_Kernel == True:
        # Ismail-Beigi and Louie's paper
        F_cart_Kernel_IBL = np.zeros((Nat, 3), dtype=complex)

else:
    F_cart_KE_IBL = np.zeros((Nat, 3))  # IBL just diag RPA
    # david thesis - diag + offdiag from kinect part
    F_cart_KE_David = np.zeros((Nat, 3))
    if Calculate_Kernel == True:
        # Ismail-Beigi and Louie's paper
        F_cart_Kernel_IBL = np.zeros((Nat, 3))

for iatom in range(Nat):
    for imode in range(Nmodes):
        F_cart_KE_IBL[iatom]   += Displacements[imode,iatom] * Sum_DKinect_diag[imode]
        F_cart_KE_David[iatom] += Displacements[imode,iatom] * (Sum_DKinect_offdiag[imode] + Sum_DKinect_diag[imode])

        if Calculate_Kernel == True: 
            F_cart_Kernel_IBL[iatom] = F_cart_Kernel_IBL[iatom] + Displacements[imode,iatom] * (Sum_DKernel[imode] + Sum_DKinect_diag[imode])
            # need to make x = x + y, instead of x += y because numpy complains that x+=y does not work when type(x)=!type(y) (one is complex and the other is real - float)


# Reporting forces in cartesian basis
DIRECTION = ['x', 'y', 'z']

arq_out = open('forces_cart.out', 'w')

print('\n\nForces (eV/ang)\n')

if Calculate_Kernel:
    header = '# Atom  dir  RPA_diag        RPA_diag_offiag RPA_diag_Kernel'
    print(header)
    arq_out.write(header + '\n')
else:
    header = '# Atom  dir  RPA_diag        RPA_diag_offiag'
    print(header)
    arq_out.write(header + '\n')

for iatom in range(Nat):
    for idir in range(3):
        if Calculate_Kernel:
            text = f'{iatom+1:<5} {DIRECTION[idir]:<5} {F_cart_KE_IBL[iatom, idir]:<15.8f} {F_cart_KE_David[iatom, idir]:<15.8f} {F_cart_Kernel_IBL[iatom, idir]:<15.8f}'
        else:
            text = f'{iatom+1:<5} {DIRECTION[idir]:<5} {F_cart_KE_IBL[iatom, idir]:<15.8f} {F_cart_KE_David[iatom, idir]:<15.8f}'
        
        print(text)
        arq_out.write(text + '\n')

arq_out.close()

end_time = time.clock_gettime(0)
report_ram()
# stopping the library
tracemalloc.stop()

print('\n\nCalculation finished!')
print(f'Total time: '+report_time(start_time))
