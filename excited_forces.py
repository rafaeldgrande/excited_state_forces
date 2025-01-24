
TESTES_DEV = False
do_vectorized_sums = True
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


def translate_bse_to_dfpt_k_points(Kpoints_in_elph_file_cart):

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

def load_el_ph_coeffs():

    # get elph coefficients from .xml files
    elph, Kpoints_in_elph_file = get_el_ph_coeffs(iq, Nirreps, dfpt_irreps_list)

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
    ikBSE_to_ikDFPT = translate_bse_to_dfpt_k_points(Kpoints_in_elph_file_cart)


    # print('DELETING ELPH')
    del elph
    report_ram()
    
    return elph_cond, elph_val, Kpoints_in_elph_file_cart, ikBSE_to_ikDFPT

def get_exciton_coeffs(iexc, jexc):
    if read_Acvk_pos == False:
        Akcv, OmegaA = get_exciton_info(exciton_file, iexc)
        Bkcv, OmegaB = get_exciton_info(exciton_file, jexc)    
    else:
        Akcv, OmegaA = get_exciton_info_alternative(Acvk_directory, iexc, Nkpoints_BSE, Ncbnds, Nvbnds)
        Bkcv, OmegaB = get_exciton_info_alternative(Acvk_directory, jexc, Nkpoints_BSE, Ncbnds, Nvbnds)

            
    # Reporting expected energies
    if iexc != jexc:
        print(f'Exciton {iexc}')
        report_expected_energies(Akcv, OmegaA)
        print(f'Exciton {jexc}')
        report_expected_energies(Bkcv, OmegaB)
    else:
        print(f'Exciton {iexc}')
        report_expected_energies(Akcv, OmegaA)

    return Akcv, OmegaA, Bkcv, OmegaB

def print_exciton_important_transitions():
    print('###############################################')
    print('Showing most relevant coeffs for this exciton')
    print('kx        ky        kz        ic   iv   abs(Acvk)^2  partial_sum(abs(Acvk)^2)')
    partial_sum = 0
    for index_Acvk in top_indexes:
        ik, ic, iv = index_Acvk
        A = Akcv[index_Acvk]
        partial_sum += abs(A)**2
        kx, ky, kz = Kpoints_BSE[ik, 0], Kpoints_BSE[ik, 1], Kpoints_BSE[ik, 2]
        print(f'{kx:8.4f}  {ky:8.4f}  {kz:8.4f}  {ic+1:<3} {iv+1:<3} {abs(A)**2:10.4f}   {partial_sum:10.6f}')    
    print('###############################################')    

def perform_kinect_sums_with_loops():

    # instead of creating big matrix, calculate sums on the fly!
    print('Creating list of indexes kcv for which sums are calculated')
    args_list_just_diag, args_list_just_offdiag = arg_lists_Dkinect(BSE_params, indexes_limited_BSE_sum)

    print("")
    Sum_DKinect_diag, Sum_DKinect_offdiag = [], []

    print('\n\nCalculating diagonal matrix elements <kcv|dH/dx_mu|kcv>')
    for imode in range(Nmodes):
        print(f"Calculating mode {imode + 1} of {Nmodes}")
        Sum_DKinect_diag.append(calc_Dkinect_matrix_simplified(Akcv, Bkcv, elph_cond, elph_val, args_list_just_diag, imode))
    
    print("\n\nCalculating off-diagonal matrix elements <kcv|dH/dx_mu|kc'v'>") 
    for imode in range(Nmodes):
        print(f"Calculating mode {imode + 1} of {Nmodes}")
        Sum_DKinect_offdiag.append(calc_Dkinect_matrix_simplified(Akcv, Bkcv, elph_cond, elph_val, args_list_just_offdiag, imode))        

    Sum_DKinect_diag, Sum_DKinect_offdiag = np.array(Sum_DKinect_diag), np.array(Sum_DKinect_offdiag)
    
    return Sum_DKinect_diag, Sum_DKinect_offdiag

def perform_kernel_sums_with_loops():
    # Kernel derivatives
    
    Sum_DKernel = np.zeros((Nmodes), dtype=np.complex64)

    if Calculate_Kernel == True:    
        EDFT = Edft_val, Edft_cond
        EQP = Eqp_val, Eqp_cond
        ELPH = elph_cond, elph_val

        DKernel = calc_deriv_Kernel((Kx+Kd)*Ry2eV, EDFT, EQP, ELPH, Akcv, Bkcv, MF_params, BSE_params)

        # dont need the kernel matrix anymore
        del Kx, Kd
        
        for imode in range(Nmodes):
            Sum_DKernel[imode] = np.sum(DKernel[imode])

    report_ram()
    
    return Sum_DKernel

########## RUNNING CODE ###################

start_time = time.clock_gettime(0)
# Getting BSE and MF parameters
# Reading eigenvecs.h5 file
get_BSE_MF_params()

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


elph_cond, elph_val, Kpoints_in_elph_file_cart, ikBSE_to_ikDFPT = load_el_ph_coeffs()

renorm_matrix_cond, renorm_matrix_val = elph_renormalization_matrix(Eqp_val, Eqp_cond, Edft_val, Edft_cond, 
                                                                    MF_params, BSE_params, ikBSE_to_ikDFPT)

if no_renorm_elph == True:
    print('Renormalizing ELPH coefficients') 
    print('where <n|dHqp|m> = <n|dHdft|m>(Eqp_n - Eqp_m)/(Edft_n - Edft_m) when Edft_n != Edft_m')
    print('and <n|dHqp|m> = <n|dHdft|m> otherwise')
    for imode in range(Nmodes):
        elph_cond[imode] = renorm_matrix_cond * elph_cond[imode]
        elph_val[imode] = renorm_matrix_val * elph_val[imode]
else:
    print('Not Renormalizing ELPH coefficients. Using <n|dHqp|m> = <n|dHdft|m> for all n and m')


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


# Getting exciton info
Akcv, OmegaA, Bkcv, OmegaB = get_exciton_coeffs(iexc, jexc)

# Getting kernel info from bsemat.h5 file
if Calculate_Kernel == True:
    Kd, Kx = get_kernel(kernel_file, factor_head)

# limited sums of BSE coefficients
indexes_limited_BSE_sum = []
if limit_BSE_sum == True:
    print('\n\nUsing limited sum of BSE coefficients. Reading transition to be used from indexes_limited_sum_BSE.dat file.')
    arq = open("indexes_limited_sum_BSE.dat")
    for line in arq:
        line_split = line.split()
        ik, ic, iv = int(line_split[0])-1, int(line_split[1])-1, int(line_split[2])-1
        indexes_limited_BSE_sum.append([ik, ic, iv])

    print('Total of transition used:', len(indexes_limited_BSE_sum))
    arq.close()


if limit_BSE_sum_up_to_value < 1.0:
    top_indexes_Akcv = summarize_Acvk(Akcv, BSE_params, limit_BSE_sum_up_to_value)
    if iexc != jexc:
        top_indexes_Bkcv = summarize_Acvk(Bkcv, BSE_params, limit_BSE_sum_up_to_value)

        # Convert lists of lists to sets of tuples
        set_A = set(tuple(item) for item in top_indexes_Akcv)
        set_B = set(tuple(item) for item in top_indexes_Bkcv)

        # Merge the sets to eliminate duplicates
        merged_set = set_A | set_B  # or set_A.union(set_B)

        # Convert the set back to a list of lists
        indexes_limited_BSE_sum = [list(item) for item in merged_set]
        
    else:
        indexes_limited_BSE_sum = top_indexes_Akcv

# summarizing Akcv information
if len(indexes_limited_BSE_sum) > 0:
    top_indexes = indexes_limited_BSE_sum
else:
    top_indexes = top_n_indexes(np.abs(Akcv), 10)

print_exciton_important_transitions()
    
########## Calculating stuff ############

if do_vectorized_sums == True:
    print("\n\nCalculating matrix elements for forces calculations <cvk|dH/dx_mu|cvk'>")
    print('!!!!!!!  Using vectorized sums for forces calculations')
    
    Sum_DKinect_diag    = np.zeros((Nmodes), dtype=np.complex64)
    Sum_DKinect_offdiag = np.zeros((Nmodes), dtype=np.complex64)
    Sum_DKernel         = np.zeros((Nmodes), dtype=np.complex64)
    
    # build A_mat[ik, ic1, ic2, iv1, iv2] = Akcv[ik, ic1, iv1] * np.conj(Bkcv[ik, ic2, iv2])
    A_mat_offdiag = Akcv[:, :, :, np.newaxis, np.newaxis] * np.conj(Bkcv[:, np.newaxis, np.newaxis, :, :])
    A_mat_diag = Akcv * np.conj(Bkcv)
    
    for imode in range(Nmodes):
        # build Gc[imode, ik, ic1, ic2, iv1, iv2] = elph_cond[imode, ik, ic1, ic2] * dirac_delta(iv1, iv2)
        # build Gv[imode, ik, ic1, ic2, iv1, iv2] = elph_val[imode, ik, iv1, iv2] * dirac_delta(ic1, ic2)
        Gc = np.zeros(A_mat_offdiag.shape, dtype=np.complex64)
        Gv = np.zeros(A_mat_offdiag.shape, dtype=np.complex64)
        
        Gc_diag, Gv_diag = np.zeros((A_mat_diag.shape), dtype=np.complex64), np.zeros((A_mat_diag.shape), dtype=np.complex64)
        
        for ik in range(Nkpoints_BSE):
            for ic in range(Ncbnds):
                for iv in range(Nvbnds):        
                    Gc_diag[ik, ic, iv] = elph_cond[imode, ik, ic, ic]
                    Gv_diag[ik, ic, iv] = elph_val[imode, ik, iv, iv]
        
        
        for ik in range(Nkpoints_BSE):
            
            for iv in range(Nvbnds):
                for ic1 in range(Ncbnds):
                    for ic2 in range(Ncbnds):
                        Gc[ik, ic1, ic2, iv, iv] = elph_cond[imode, ik, ic1, ic2]
            
            for ic in range(Ncbnds):
                for iv1 in range(Nvbnds):
                    for iv2 in range(Nvbnds):
                        Gv[ik, ic, ic, iv1, iv2] = elph_val[imode, ik, iv1, iv2]
        
        # Multiply A_mat * (Gc[imode] - Gv[imode])
        
        F_imode = A_mat_offdiag * (Gc - Gv)
        
        # F_imode_diag = np.diagonal(F_imode, axis1=1, axis2=2)
        # F_imode_diag = np.diagonal(F_imode_diag, axis1=2, axis2=3)  # Now shape (nk, nc, nv)
        F_imode_diag = A_mat_diag * (Gc_diag - Gv_diag)

        Sum_DKinect_offdiag[imode] = np.sum(F_imode)
        Sum_DKinect_diag[imode] = np.sum(F_imode_diag)
        
    Sum_DKinect_offdiag = Sum_DKinect_offdiag - Sum_DKinect_diag
    
else:
    print("\n\nCalculating matrix elements for forces calculations <cvk|dH/dx_mu|c'v'k'>")
    Sum_DKinect_diag, Sum_DKinect_offdiag = perform_kinect_sums_with_loops()
    Sum_DKernel = perform_kernel_sums_with_loops()

# Convert from Ry/bohr to eV/A. Minus sign comes from F=-dV/du

Sum_DKinect_diag    = -Sum_DKinect_diag * Ry2eV / bohr2A
Sum_DKinect_offdiag = -Sum_DKinect_offdiag * Ry2eV / bohr2A
Sum_DKernel         = -Sum_DKernel * Ry2eV / bohr2A

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
