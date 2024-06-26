
TESTES_DEV = False
verbosity = 'high'

run_parallel = False
if run_parallel == True:
    from multiprocessing import Pool
    from multiprocessing import freeze_support



# FIRST MESSAGE

# excited state forces modules
from excited_forces_config import *
from bgw_interface_m import *
from qe_interface_m import *
from excited_forces_m import *

import numpy as np

import tracemalloc  # track ram usage
tracemalloc.start()

import time
from datetime import datetime
# datetime object containing current date and time
now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("\n\nExecution date: ", dt_string)

print('\n\n*************************************************************')
print('Excited state forces code')
print('Developed by Rafael Del Grande and David Strubbe')
print('*************************************************************\n\n')


# Report functions
def report_time(start_time):
    end_time_func = time.clock_gettime(0)
    text = f'{(end_time_func - start_time)/60:.2f} min'
    return text


def report_ram():
    temp_ram = tracemalloc.get_traced_memory()[0] / 1024**2
    max_temp_ram = tracemalloc.get_traced_memory()[1] / 1024**2

    print('\n\n############### RAM REPORT #################')
    print(f'RAM used now: {temp_ram:.2f} MB')
    print(f'Max RAM used until now: {max_temp_ram:.2f} MB')
    print('############################################\n\n')

# Classes


class Parameters_BSE:

    def __init__(self, Nkpoints_BSE, Kpoints_BSE, Ncbnds, Nvbnds, Nval, Ncbnds_sum, Nvbnds_sum, Ncbnds_coarse, Nvbnds_coarse, Nkpoints_coarse, rec_cell_vecs):
        self.Nkpoints_BSE = Nkpoints_BSE
        self.Kpoints_BSE = Kpoints_BSE
        self.Ncbnds = Ncbnds
        self.Nvbnds = Nvbnds
        self.Nval = Nval
        self.Ncbnds_sum = Ncbnds_sum
        self.Nvbnds_sum = Nvbnds_sum
        self.Ncbnds_coarse = Ncbnds_coarse
        self.Nvbnds_coarse = Nvbnds_coarse
        self.Nkpoints_coarse = Nkpoints_coarse
        self.rec_cell_vecs = rec_cell_vecs

class Parameters_MF:

    def __init__(self, Nat, atomic_pos, cell_vecs, cell_vol, alat):
        self.Nat = Nat
        self.atomic_pos = atomic_pos
        self.cell_vecs = cell_vecs
        self.cell_vol = cell_vol
        self.alat = alat
        self.Nmodes = 3 * Nat


class Parameters_ELPH:

    def __init__(self, Nkpoints_DPFT, Kpoints_DFPT):
        self.Nkpoints_DFPT = Nkpoints_DPFT
        self.Kpoints_DFPT = Kpoints_DFPT

# functions


def get_BSE_MF_params():

    global MF_params, BSE_params, Nmodes
    global Nat, atomic_pos, cell_vecs, cell_vol, alat
    global Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval
    global Nvbnds_sum, Ncbnds_sum
    global Nvbnds_coarse, Ncbnds_coarse, Nkpoints_coarse
    global rec_cell_vecs, Nmodes

    if read_Acvk_pos == False:
        Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs = get_params_from_eigenvecs_file(exciton_file)
    else:
        Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs = get_params_from_alternative_file('params')
    
    Nmodes = 3 * Nat

    if 0 < ncbnds_sum < Ncbnds:
        print('*********************************')
        print('Instead of using all cond bands from the BSE hamiltonian')
        print(f'I will use {ncbnds_sum} cond bands (variable ncbnds_sum)')
        print('*********************************')
        Ncbnds_sum = ncbnds_sum
    else:
        Ncbnds_sum = Ncbnds

    if 0 < nvbnds_sum < Nvbnds:
        print('*********************************')
        print('Instead of using all val bands from the BSE hamiltonian')
        print(f'I will use {nvbnds_sum} val bands (variable nvbnds_sum)')
        print('*********************************')
        Nvbnds_sum = nvbnds_sum
    else:
        Nvbnds_sum = Nvbnds
        
    if elph_fine_a_la_bgw == True:
        print('I will perform elph interpolation "a la BerkeleyGW"')
        print('Check the absorption.inp file to see how many bands were used in both coarse and fine grids.')
        print('From the forces.inp file, I got the following parameters: ')
        print(f'    ncond_coarse    = {ncbands_co}')
        print(f'    nval_coarse     = {nvbands_co}')
        print(f'    nkpoints_coarse = {nkpnts_co}')
        print('Be sure that all those bands are included in the DFPT calculation!')
        print('If not, the missing elph coefficients will be considered to be equal 0.')
        
    Ncbnds_coarse = ncbands_co
    Nvbnds_coarse = nvbands_co
    Nkpoints_coarse = nkpnts_co
        

    MF_params = Parameters_MF(Nat, atomic_pos, cell_vecs, cell_vol, alat)
    BSE_params = Parameters_BSE(Nkpoints_BSE, Kpoints_BSE, Ncbnds, Nvbnds, Nval, Ncbnds_sum, Nvbnds_sum, Ncbnds_coarse, Nvbnds_coarse, Nkpoints_coarse, rec_cell_vecs)

def report_expected_energies(Akcv, Omega):

    Mean_Ekin = 0.0
    if Calculate_Kernel == True:
        Mean_Kx, Mean_Kd = 0.0, 0.0

    for ik1 in range(BSE_params.Nkpoints_BSE):
        for ic1 in range(BSE_params.Ncbnds):
            for iv1 in range(BSE_params.Nvbnds):
                Mean_Ekin += (Eqp_cond[ik1, ic1] - Eqp_val[ik1, iv1])*abs(Akcv[ik1, ic1, iv1])**2

    if Calculate_Kernel == True:
        for ik1 in range(BSE_params.Nkpoints_BSE):
            for ic1 in range(BSE_params.Ncbnds):
                for iv1 in range(BSE_params.Nvbnds):
                    for ik2 in range(BSE_params.Nkpoints_BSE):
                        for ic2 in range(BSE_params.Ncbnds):
                            for iv2 in range(BSE_params.Nvbnds):
                                Mean_Kx += Ry2eV * \
                                    np.conj(
                                        Akcv[ik1, ic1, iv1]) * Kx[ik2, ik1, ic2, ic1, iv2, iv1] * Akcv[ik2, ic2, iv2]
                                Mean_Kd += Ry2eV * \
                                    np.conj(
                                        Akcv[ik1, ic1, iv1]) * Kd[ik2, ik1, ic2, ic1, iv2, iv1] * Akcv[ik2, ic2, iv2]

    print('Exciton energies (eV): ')
    print(f'    Omega         =  {Omega:.6f}')
    print(f'    <KE>          =  {Mean_Ekin:.6f}')
    print(f'    Omega - <KE>  =  {(Omega - Mean_Ekin):.6f}')
    if Calculate_Kernel == True:
        print(f'    <Kx>          =  {np.real(Mean_Kx):.6f} + {np.imag(Mean_Kx):.6f} j')
        print(f'    <Kd>          =  {np.real(Mean_Kd):.6f} + {np.imag(Mean_Kd):.6f} j')
        DIFF = Omega - (Mean_Ekin + Mean_Kd + Mean_Kx)
        print(f'\n    DIFF          = {DIFF:.6f} \n\n')


def correct_comp_vector(comp):
    # component is in alat units
    # return the component in the interval 0 < comp < 1
    
    # making -1 < comp < 1
    comp = round(comp, 6) - int(round(comp, 6))
    if comp < 0: # making comp 0 < comp < 1
        comp += 1

    return comp


def find_kpoint(kpoint, K_list):
    index_in_matrix = -1
    for index in range(len(K_list)):
        # if np.array_equal(kpoint, K_list[index]):
        if np.linalg.norm(kpoint - K_list[index]) <= TOL_DEG:
            index_in_matrix = index
    return index_in_matrix


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


########## RUNNING CODE ###################

start_time = time.clock_gettime(0)
# Getting BSE and MF parameters
# Reading eigenvecs.h5 file
get_BSE_MF_params()

# Getting exciton info
if read_Acvk_pos == False:
    Akcv, OmegaA = get_exciton_info(exciton_file, iexc)
    if iexc != jexc:  
        Bkcv, OmegaB = get_exciton_info(exciton_file, jexc)
    else:
        Bkcv, OmegaB = Akcv, OmegaA
        
else:
    Akcv, OmegaA = get_exciton_info_alternative(Acvk_directory, iexc, Nkpoints_BSE, Ncbnds, Nvbnds)
    if iexc != jexc:
        Bkcv, OmegaB = get_exciton_info_alternative(Acvk_directory, jexc, Nkpoints_BSE, Ncbnds, Nvbnds)
    else:
        Bkcv, OmegaB = Akcv, OmegaA

# Printing relevant information for this exciton
index_of_max_abs_value_Akcv = summarize_Acvk(Akcv, BSE_params)
if iexc != jexc:
    index_of_max_abs_value_Bkcv = summarize_Acvk(Bkcv, BSE_params)

# getting info from eqp.dat (from absorption calculation)
Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, BSE_params)

# Getting kernel info from bsemat.h5 file
if Calculate_Kernel == True:
    Kd, Kx = get_kernel(kernel_file, factor_head)

# Reporting expected energies
if iexc != jexc:
    print(f'Exciton {iexc}')
    report_expected_energies(Akcv, OmegaA)
    print(f'Exciton {jexc}')
    report_expected_energies(Bkcv, OmegaB)
else:
    print(f'Exciton {iexc}')
    report_expected_energies(Akcv, OmegaA)

# Getting elph coefficients

# get displacement patterns
iq = 0  # FIXME -> generalize for set of q points. used for excitons with non-zero center of mass momentum
Displacements, Nirreps = get_patterns2(iq, MF_params)

# get elph coefficients from .xml files
# if run_parallel == False:

elph, Kpoints_in_elph_file = get_el_ph_coeffs(iq, Nirreps)
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

# apply acoustic sum rule
# elph = impose_ASR(elph, Displacements, MF_params, acoutic_sum_rule)
# print('!!!!!! SHAPE', np.shape(Eqp_val))

# filter data to get just g_c1c2 and g_v1v2
elph_cond, elph_val = filter_elph_coeffs(elph, MF_params, BSE_params) 

# impose acoustic sum rule over elph coefficients
elph_cond = impose_ASR(elph_cond, Displacements, MF_params, acoutic_sum_rule)
elph_val = impose_ASR(elph_val, Displacements, MF_params, acoutic_sum_rule)

# Let's put all k points from BSE grid in the first Brillouin zone
ikBSE_to_ikDFPT = translate_bse_to_dfpt_k_points()


# summarize transition energies and derivatives of (Ec-Ev)
# index_of_max_abs_value_Akcv
ik, ic, iv = index_of_max_abs_value_Akcv
Emin_gap_dft = Edft_cond[ik, ic] - Edft_val[ik, ic]
Emin_gap_qp = Eqp_cond[ik, ic] - Eqp_val[ik, ic]
print(f'Highest value of Acvk for exciton {iexc}')
print(f'occurs at k point {Kpoints_BSE[ik][0]:4f}  {Kpoints_BSE[ik][1]:4f}  {Kpoints_BSE[ik][2]:4f}')
print(f'At this point the gap is equal to')
print(f'at DFT level: {Emin_gap_dft} eV')
print(f'at GW level:  {Emin_gap_qp} eV')

# use modified Acvk

if use_Acvk_single_transition == True:
    print('WARNING! use_Acvk_single_transition flag is True')
    print('We are making new_Acvk = delta(c0v0k0,c0v0k0), where c0v0k0 is the transition where Acvk is maximum')
    ik, ic, iv = index_of_max_abs_value_Akcv
    Akcv = np.zeros(Akcv.shape, dtype=complex)
    Akcv[ik, ic, iv] = 1.0 + 0.0j
    if iexc != jexc:
        Bkcv = np.zeros(Akcv.shape, dtype=complex)
        Bkcv[ik, ic, iv] = 1.0 + 0.0j     



print('Derivatives for Emin gap for different modes')
# this diagonal matrix element is the same at qp and dft levels in our approximation 
der_E_gap_dr = elph_cond[:, ik, ic, ic] - elph_val[:, ik, ic, ic]
max_der_E_gap_dr = np.max(np.abs(der_E_gap_dr))
for imode in range(Nmodes):
    if np.abs(der_E_gap_dr[imode]) >= max_der_E_gap_dr * 0.1:
        print(f'mode {imode} : {der_E_gap_dr[imode]:.8f} eV/angs')


# report_ram()
# print('DELETING ELPH')
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

    
    
########## Calculating stuff ############

print("Calculating matrix elements for forces calculations <cvk|dH/dx_mu|c'v'k'>")

# creating KCV list with indexes ik, ic, iv (in this order) used to vectorize future sums
KCV_list = []

for ik in range(BSE_params.Nkpoints_BSE):
    for ic in range(BSE_params.Ncbnds_sum):
        for iv in range(BSE_params.Nvbnds_sum):
            KCV_list.append((ik, ic, iv))            


# Creating auxialiry quantities
# aux_cond_matrix[imode, ik, ic1, ic2] = elph_cond[imode, ik, ic1, ic2] * deltaEqp / deltaEdft (if ic1 != ic2)
# aux_val_matrix[imode, ik, iv1, iv2]  = elph_val[imode, ik, iv1, iv2]  * deltaEqp / deltaEdft (if iv1 != iv2)
# If ic1 == ic2 (iv1 == iv2), then the matrix elements are just the elph coefficients"""
aux_cond_matrix, aux_val_matrix = aux_matrix_elem(
    elph_cond, elph_val, Eqp_val, Eqp_cond, Edft_val, Edft_cond, MF_params, BSE_params, ikBSE_to_ikDFPT)

# Calculating matrix elements F_cvkc'v'k'
# DKinect = calc_Dkinect_matrix(
#     Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, MF_params, BSE_params)

# instead of creating big matrix, calculate sums on the fly!

args_list_just_diag, args_list_just_offdiag = arg_lists_Dkinect(BSE_params)

Sum_DKinect_diag, Sum_DKinect_just_offdiag = [], []

if run_parallel == False:
    
##### test - run in a "dummy" way

    # if __name__ == '__main__':
    #     # from multiprocessing import Pool, cpu_count
    #     from functools import partial
        
    #     # num_processes = 1 # cpu_count()

    #     print('STEP 1')
        
    #     for imode in range(Nmodes):
    #     # Create a partial function with fixed arguments using functools.partial
    #         partial_func = partial(calc_Dkinect_matrix_elem, Akcv=Akcv, Bkcv=Bkcv, aux_cond_matrix=aux_cond_matrix, aux_val_matrix=aux_val_matrix, imode=imode)

    #         print('STEP 3')
    #         # Create a Pool object for parallel processing
    #         with Pool(processes=num_processes) as pool:
    #             # Use pool.map to apply the partial function to the args_list in parallel
    #             results_just_diag = pool.map(partial_func, args_list_just_diag)
    #             results_just_offdiag = pool.map(partial_func, args_list_just_offdiag)
                
    #         print('STEP 4')
    #         # Compute the sum of the results
    #         Sum_DKinect_diag.append(np.sum(np.array(results_just_diag)))
    #         Sum_DKinect_just_offdiag.append(np.sum(np.array(results_just_offdiag)))
        

    Sum_DKinect_diag, Sum_DKinect = [], []
    for imode in range(Nmodes):
        Sum_DKinect_diag.append(calc_Dkinect_matrix_simplified(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, args_list_just_diag, imode))
        Sum_DKinect.append(calc_Dkinect_matrix_simplified(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, args_list_just_offdiag, imode))        
    Sum_DKinect_diag, Sum_DKinect = np.array(Sum_DKinect_diag), np.array(Sum_DKinect)

else:
    if __name__ == '__main__':
        # import multiprocessing
        
        Sum_DKinect_diag, Sum_DKinect = [], []
        for imode in range(Nmodes):
            Sum_DKinect_diag.append(calc_Dkinect_matrix_parallel(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, args_list_just_diag, imode))
            Sum_DKinect.append(calc_Dkinect_matrix_parallel(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, args_list_just_offdiag, imode))
            
        Sum_DKinect_diag, Sum_DKinect = np.array(Sum_DKinect_diag), np.array(Sum_DKinect)

    
if TESTES_DEV == True:
    Sum_DKinect_diag_test, Sum_DKinect_test = calc_Dkinect_matrix_ver2_master(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, MF_params, BSE_params, KCV_list, run_parallel)
    
# print(Sum_DKinect_diag, type(Sum_DKinect_diag))

# arq_f_disp = open('forces_vs_displacement.dat', 'w')
# arq_f_disp.write('########## Forces in (eV/A) ################\n')
# arq_f_disp.write('# disp_pattern F_diag F_diag_offdiag\n')

# for imode in range(Nmodes):

#     sum_temp = 0.0 + 0.0j
#     for ik in range(Nkpoints_BSE):
#         for ic in range(Ncbnds):
#             for iv in range(Nvbnds):
#                 sum_temp += DKinect[imode, ik, ic, iv, ik, ic, iv]

#     Sum_DKinect_diag[imode] = sum_temp

#     # sum of off-diagonal part + sum of diagonal part
#     Sum_DKinect[imode] = np.sum(DKinect[imode])

#     arq_f_disp.write(
#         f'{imode+1}   {Sum_DKinect_diag[imode]}    {Sum_DKinect[imode] }\n')


# arq_f_disp.close()

del aux_cond_matrix, aux_val_matrix

# Kernel derivatives
if Calculate_Kernel == True:

    EDFT = Edft_val, Edft_cond
    EQP = Eqp_val, Eqp_cond
    ELPH = elph_cond, elph_val

    DKernel = calc_deriv_Kernel((Kx+Kd)*Ry2eV, EDFT, EQP, ELPH, Akcv, Bkcv, MF_params, BSE_params)

    del Kx, Kd

print("Calculating sums")

if Calculate_Kernel == True:
    Sum_DKernel = np.zeros((Nmodes), dtype=np.complex64)

    for imode in range(Nmodes):
        Sum_DKernel[imode] = np.sum(DKernel[imode])

report_ram()

# Convert from Ry/bohr to eV/A. Minus sign comes from F=-dV/du

Sum_DKinect_diag = -Sum_DKinect_diag*Ry2eV/bohr2A
Sum_DKinect      = -Sum_DKinect*Ry2eV/bohr2A

if Calculate_Kernel == True:
    Sum_DKernel = -Sum_DKernel*Ry2eV/bohr2A

# Warn if imag part is too big (>= 10^-6)

if max(abs(np.imag(Sum_DKinect_diag))) >= 1e-6:
    print('WARNING: Imaginary part of kinectic diagonal forces >= 10^-6 eV/angs!')

if max(abs(np.imag(Sum_DKinect))) >= 1e-6:
    print('WARNING: Imaginary part of kinectic forces >= 10^-6 eV/angs!')

if Calculate_Kernel == True:
    if max(abs(np.imag(Sum_DKernel))) >= 1e-6:
        print('WARNING: Imaginary part of Kernel forces >= 10^-6 eV/angs!')

# Show just real part of numbers (default)

                                                                                                                                                                                                                                                                                                                                    
if show_imag_part == False:
    Sum_DKinect_diag = np.real(Sum_DKinect_diag)
    Sum_DKinect = np.real(Sum_DKinect)

    if Calculate_Kernel == True:
        Sum_DKernel = np.real(Sum_DKernel)

# Calculate forces cartesian basis

print("Calculating forces in cartesian basis")

if show_imag_part == True:
    F_cart_KE_IBL = np.zeros((Nat, 3), dtype=complex)  # IBL just diag RPA
    # david thesis - diag + offdiag from kinect part
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
        F_cart_KE_IBL[iatom] += Displacements[imode,
                                              iatom] * Sum_DKinect_diag[imode]
        F_cart_KE_David[iatom] += Displacements[imode,
                                                iatom] * Sum_DKinect[imode]

        if Calculate_Kernel == True:
            F_cart_Kernel_IBL[iatom] = F_cart_Kernel_IBL[iatom] + Displacements[imode,
                                                                                iatom] * (Sum_DKernel[imode] + Sum_DKinect_diag[imode])

            # need to make x = x + y, instead of x += y because numpy complains that x+=y does not work when type(x)=!type(y) (one is complex and the other is real - float)


# Reporting forces in cartesian basis
DIRECTION = ['x', 'y', 'z']

arq_out = open('forces_cart.out', 'w')

print('\n\nForces (eV/ang)\n')

if Calculate_Kernel == True:
    print('# Atom  dir  RPA_diag RPA_diag_offiag RPA_diag_Kernel')
    arq_out.write('# Atom  dir  RPA_diag RPA_diag_offiag RPA_diag_Kernel \n')
else:
    print('# Atom  dir  RPA_diag RPA_diag_offiag ')
    arq_out.write('# Atom  dir  RPA_diag RPA_diag_offiag \n')

for iatom in range(Nat):
    for idir in range(3):
        text = f'{iatom+1}  {DIRECTION[idir]}  {F_cart_KE_IBL[iatom, idir]:.8f}  {F_cart_KE_David[iatom, idir]:.8f}'
        if Calculate_Kernel == True:
            text += f'  {F_cart_Kernel_IBL[iatom, idir]:.8f} '

        print(text)
        arq_out.write(text+'\n')

arq_out.close()

end_time = time.clock_gettime(0)
report_ram()
# stopping the library
tracemalloc.stop()

print('\n\nCalculation finished!')
print(f'Total time: '+report_time(start_time))
