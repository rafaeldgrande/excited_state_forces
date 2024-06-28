

# TODO -> splits this module for the different functions
#          -> reads GW/BSE data
#          -> reads DFPT data
#          -> calculates stuff
#          -> etc


import numpy as np
import xml.etree.ElementTree as ET
import h5py
from datetime import datetime


from excited_forces_config import *

if run_parallel == True:
    from multiprocessing import Pool, cpu_count
    from functools import partial



def calc_DKernel_mat_elem(indexes, Kernel, EDFT, EQP, ELPH, MF_params, BSE_params):
    """Calculates derivatives of kernel matrix elements"""

    ik1, ik2, iv1, iv2, ic1, ic2, imode = indexes

    elph_cond, elph_val = ELPH
    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds_sum = BSE_params.Ncbnds_sum
    Nvbnds_sum = BSE_params.Nvbnds_sum
    Edft_val, Edft_cond = EDFT
    Eqp_val, Eqp_cond = EQP

    DKelement = 0 + 0.0*1.0j    

    # ik2, ik1, ic2, ic1, iv2, iv1

    for ivp in range(Nvbnds_sum):

        DeltaEdft = Edft_val[ik1, iv1] - Edft_val[ik1, ivp]
        DeltaEqp = Eqp_val[ik1, iv1] - Eqp_val[ik1, ivp]

        indexes_K = ik2, ik1, ic2, ic1, iv2, ivp
        indexes_g = imode, ik1, ivp, iv1

        if abs(DeltaEdft) > TOL_DEG:
            if no_renorm_elph == True:
                DKelement += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEqp
            else:  # assuming that gw levels have the same degeneracy of dft levels
                DKelement += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEdft

        DeltaEdft = Edft_val[ik2, iv2] - Edft_val[ik2, ivp]
        DeltaEqp = Eqp_val[ik2, iv2] - Eqp_val[ik2, ivp]

        indexes_K = ik2, ik1, ic2, ic1, ivp, iv1
        indexes_g = imode, ik2, iv2, ivp

        if abs(DeltaEdft) > TOL_DEG:
            if no_renorm_elph == True:
                DKelement += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEqp
            else:
                DKelement += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEdft

    for icp in range(Ncbnds_sum):

        DeltaEdft = Edft_cond[ik1, ic1] - Edft_cond[ik1, icp]
        DeltaEqp = Eqp_cond[ik1, ic1] - Eqp_cond[ik1, icp]

        indexes_K = ik2, ik1, ic2, icp, iv2, iv1
        indexes_g = imode, ik1, icp, ic1

        if abs(DeltaEdft) > TOL_DEG:
            if no_renorm_elph == True:
                DKelement += Kernel[indexes_K]*elph_cond[indexes_g]/DeltaEqp
            else:
                DKelement += Kernel[indexes_K] * elph_cond[indexes_g]/DeltaEdft

        DeltaEdft = Edft_cond[ik2, ic2] - Edft_cond[ik2, icp]
        DeltaEqp = Eqp_cond[ik2, ic2] - Eqp_cond[ik2, icp]

        indexes_K = ik2, ik1, icp, ic1, iv2, iv1
        indexes_g = imode, ik2, ic2, icp

        if abs(DeltaEdft) > TOL_DEG:
            if no_renorm_elph == True:
                DKelement += Kernel[indexes_K]*elph_cond[indexes_g]/DeltaEqp
            else:
                DKelement += Kernel[indexes_K] * elph_cond[indexes_g]/DeltaEdft

    return DKelement


def calc_deriv_Kernel(KernelMat, EDFT, EQP, ELPH, Akcv, Bkcv, MF_params, BSE_params):

    print("\n\n    - Calculating Kernel part")
    
    if write_dK_mat == True:
        print('Writing kernel derivatives in the file dK_mat.dat')
        arq_dK_report = open('dK_mat.dat', 'w')
        arq_dK_report.write('# imode ik1 ik2 iv1 iv2 ic1 ic2 <k1v1c1|d_(mode) K|k2v2c2> \n')

    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds_sum = BSE_params.Ncbnds_sum
    Nvbnds_sum = BSE_params.Nvbnds_sum

    # Shape_kernel = (Nmodes, Nkpoints, Ncbnds, Nvbnds, Nkpoints, Ncbnds, Nvbnds)
    # DKernel = np.zeros(Shape_kernel, dtype=np.complex64)
    # DKernel_IBL = np.zeros(Shape_kernel, dtype=np.complex64)
    Sum_DKernel = np.zeros((Nmodes), dtype=np.complex64)
    Sum_DKernel_IBL = np.zeros((Nmodes), dtype=np.complex64)

    total_ite = Nmodes*(Nkpoints*Ncbnds_sum*Nvbnds_sum)**2
    interval_report = max(int(np.sqrt(total_ite)), int(total_ite/20))
    i_term = 0
    print(f'        Total terms to be calculated : {total_ite}')

    for imode in range(Nmodes):

        temp_sum_imode     = 0.0 + 0.0j

        for ik1 in range(Nkpoints):
            for ic1 in range(Ncbnds_sum):
                for iv1 in range(Nvbnds_sum):

                    A_bra = np.conj(Akcv[ik1, ic1, iv1])

                    for ik2 in range(Nkpoints):
                        for ic2 in range(Ncbnds_sum):
                            for iv2 in range(Nvbnds_sum):

                                B_ket = Bkcv[ik2, ic2, iv2]

                                indexes = ik1, ik2, iv1, iv2, ic1, ic2, imode
                                dK = calc_DKernel_mat_elem(
                                    indexes, KernelMat, EDFT, EQP, ELPH, MF_params, BSE_params)
                                temp_sum_imode += A_bra * dK * B_ket
                                    # DKernel[imode, ik1, ic1, iv1, ik2,
                                    #         ic2, iv2] = A_bra * dK * A_ket

                                i_term += 1
                                if i_term % interval_report == 0:
                                    print(
                                        f'        {i_term} of {total_ite} calculated --------- {round(100*i_term/total_ite,1)} %')

                                if write_dK_mat == True:
                                    arq_dK_report.write(f'{imode} {ik1+1} {ik2+1} {iv1+1} {iv2+1} {ic1+1} {ic2+1} {A_bra * dK * B_ket}\n')
                                
        Sum_DKernel[imode] = temp_sum_imode

    if write_dK_mat == True:
        arq_dK_report.close()
    
    return Sum_DKernel
    

def aux_matrix_elem(elph_cond, elph_val, Eqp_val, Eqp_cond, Edft_val, Edft_cond, MF_params, BSE_params, ikBSE_to_ikDFPT):
    """ Calculates auxiliar matrix elements to be used later in the forces matrix elements.
    Returns aux_cond_matrix, aux_val_matrix and
    aux_cond_matrix[imode, ik, ic1, ic2] = elph_cond[imode, ik, ic1, ic2] * deltaEqp / deltaEdft (if ic1 != ic2)
    aux_val_matrix[imode, ik, iv1, iv2]  = elph_val[imode, ik, iv1, iv2]  * deltaEqp / deltaEdft (if iv1 != iv2)
    If deltaEdft == 0, then the matrix elements are just the elph coefficients"""

    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds_sum = BSE_params.Ncbnds_sum
    Nvbnds_sum = BSE_params.Nvbnds_sum

    Shape_cond = (Nmodes, Nkpoints, Ncbnds_sum, Ncbnds_sum)
    aux_cond_matrix = np.zeros(Shape_cond, dtype=np.complex64)

    Shape_val = (Nmodes, Nkpoints, Nvbnds_sum, Nvbnds_sum)
    aux_val_matrix = np.zeros(Shape_val, dtype=np.complex64)

    for imode in range(Nmodes):
        for ik in range(Nkpoints):

            ik_dfpt = ikBSE_to_ikDFPT[ik]

            if ik_dfpt != -1:
                # remember that order of k points in DFPT file
                # is not always equal to the order in the eigenvecs file
                # -1 means that the code did not find the corresponce
                # between the 2 kgrids

                for ic1 in range(Ncbnds_sum):
                    for ic2 in range(Ncbnds_sum):

                        elph = elph_cond[imode, ik_dfpt, ic1, ic2]

                        if no_renorm_elph == True:
                            aux_cond_matrix[imode, ik, ic1, ic2] = elph

                        else:
                            deltaEqp = Eqp_cond[ik, ic1] - Eqp_cond[ik, ic2]
                            deltaEdft = Edft_cond[ik, ic1] - Edft_cond[ik, ic2]

                            if abs(Edft_cond[ik, ic1] - Edft_cond[ik, ic2]) <= TOL_DEG:
                                aux_cond_matrix[imode, ik, ic1, ic2] = elph
                            else:
                                aux_cond_matrix[imode, ik, ic1, ic2] = elph * deltaEqp / deltaEdft
                            

                for iv1 in range(Nvbnds_sum):
                    for iv2 in range(Nvbnds_sum):

                        elph = elph_val[imode, ik_dfpt, iv1, iv2]

                        if no_renorm_elph == True:
                            aux_val_matrix[imode, ik, iv1, iv2] = elph

                        else:
                            deltaEqp = Eqp_val[ik, iv1] - Eqp_val[ik, iv2]
                            deltaEdft = Edft_val[ik, iv1] - Edft_val[ik, iv2]

                            if abs(Edft_val[ik, iv1] - Edft_val[ik, iv2]) <= TOL_DEG:
                                aux_val_matrix[imode, ik, iv1, iv2] = elph
                            else:
                                aux_val_matrix[imode, ik, iv1, iv2] = elph * deltaEqp / deltaEdft
            else:
                print(f'Kpoint {ik} not found. Skipping the calculation for this k point')

    return aux_cond_matrix, aux_val_matrix


def delta(i, j):
    """Dirac delta - TODO: check if python has a builtin function that does that."""
    if i == j:
        return 1.0
    else:
        return 0.0


def dirac_delta_Edft(i, j, Edft, TOL_DEG):
    """Dirac delta function in energy space based on the energy difference between two states.

    Args:
        i (int): index of the first state
        j (int): index of the second state
        Edft (ndarray): array of energies in eV
        TOL_DEG (float): energy tolerance in eV

    Returns:
        float: 1.0 if |E_i - E_j| < TOL_DEG, 0.0 otherwise
    """

    energy_diff = abs(Edft[0, i] - Edft[0, j])
    
    if energy_diff > TOL_DEG:
        return 1.0
    else:
        return 0.0


def calc_Dkinect_matrix_elem(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2):
    """Calculates excited state force matrix elements."""

    # calculate matrix element imode, ik, ic1, ic2, iv1, iv2
    temp_cond = aux_cond_matrix[imode, ik, ic1, ic2] * delta(iv1, iv2)
    temp_val = aux_val_matrix[imode, ik, iv1, iv2] * delta(ic1, ic2)

    tempA = Akcv[ik, ic1, iv1] * np.conj(Bkcv[ik, ic2, iv2])
    Dkin = tempA * (temp_cond - temp_val)

    if report_RPA_data == True:
        arq_RPA_data = open('RPA_matrix_elements.dat', 'a')
        temp_text = f'{imode+1} {ik+1} {ic1+1} {ic2+1} {iv1+1} {iv2+1} {Dkin} {tempA} {temp_cond} {temp_val} \n'
        arq_RPA_data.write(temp_text)
        arq_RPA_data.close()

    return Dkin

def calc_Dkinect_matrix(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, MF_params, BSE_params):

    now_this_func = datetime.now()

    print("\n\n     - Calculating RPA part")

    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds_sum = BSE_params.Ncbnds_sum
    Nvbnds_sum = BSE_params.Nvbnds_sum

    Sum_DKinect_diag = np.zeros((Nmodes), dtype=complex)
    Sum_DKinect      = np.zeros((Nmodes), dtype=complex)

    if report_RPA_data == True:
        arq_RPA_data = open('RPA_matrix_elements.dat', 'w')
        arq_RPA_data.write(
            '# mode ik ic1 ic2 iv1 iv2 F conj(Akc1v1)*Akc2v2 auxMatcond(c1,c2) auxMatval(v1,v2)\n')
        arq_RPA_data.close()

    if just_RPA_diag == True:
        
        print('Calculating just diag RPA force matrix elements')

        # reporting
        total_iterations = Nmodes*Nkpoints*Ncbnds_sum*Nvbnds_sum
        interval_report = step_report(total_iterations)
        counter = 0
        print(f'Total terms to be calculated : {total_iterations}')     

        for imode in range(Nmodes):
            temp_imode_just_diag = 0.0 + 0.0j

            for ik in range(Nkpoints):
                for ic1 in range(Ncbnds_sum):
                    for iv1 in range(Nvbnds_sum):
                        temp = calc_Dkinect_matrix_elem(
                            Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic1, iv1, iv1)
                        # DKinect[imode, ik, ic1, iv1, ik, ic1, iv1] = temp
                        temp_imode_just_diag += temp 

                        # reporting
                        counter += 1
                        report_iterations(counter, total_iterations, interval_report, now_this_func)


            Sum_DKinect_diag[imode] = temp_imode_just_diag

    else:

        print('Calculating diag and offdiag RPA force matrix elements')

        if use_hermicity_F == False:

            # reporting
            total_iterations = Nmodes*Nkpoints*(Ncbnds_sum*Nvbnds_sum)**2
            interval_report = step_report(total_iterations)
            counter = 0
            print(f'Total terms to be calculated : {total_iterations}')

            for imode in range(Nmodes):
                temp_imode = 0.0 + 0.0j
                temp_imode_just_diag = 0.0 + 0.0j

                for ik in range(Nkpoints):
                    for ic1 in range(Ncbnds_sum):
                        for iv1 in range(Nvbnds_sum):
                            temp = calc_Dkinect_matrix_elem(
                                Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic1, iv1, iv1)
                            temp_imode_just_diag += temp
                            for ic2 in range(Ncbnds_sum):
                                for iv2 in range(Nvbnds_sum):
                                    temp = calc_Dkinect_matrix_elem(
                                        Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2)
                                    temp_imode += temp
                                    # DKinect[imode, ik, ic1, iv1,
                                            # ik, ic2, iv2] = temp

                                    # reporting
                                    counter += 1
                                    report_iterations(counter, total_iterations, interval_report, now_this_func)

                Sum_DKinect_diag[imode] = temp_imode_just_diag
                Sum_DKinect[imode]      = temp_imode

        # Creating a list of tuples with cond and val bands indexes
        # [(0,0), (0,1), (0,2), ... (Ncbnds-1, 0), (Ncbnds-1, 1), ..., (Ncbnds-1, Nvbnds-1)]
        # size of this list Nvbnds*Ncbnds

        # New block - now using the fact that F_cvc'v' = conj(F_c'v'cv)
        # Reduces the number of computed terms by about half

        # Cannot use this if iexc != jexc 
        # If use_hermicity_F == True and iexc != jexc
        # then I just make use_hermicity_F == False

        else:

            print('Using "hermicity" of force matrix elements')

            indexes_cv = [(icp, ivp) for ivp in range(Nvbnds_sum)
                          for icp in range(Ncbnds_sum)]

            # reporting
            total_iterations = int(Nmodes*Nkpoints*len(indexes_cv)*(len(indexes_cv)+1) / 2)
            interval_report = step_report(total_iterations)
            counter = 0
            print(f'Total terms to be calculated : {total_iterations}')

            for imode in range(Nmodes):
                temp_imode_not_diag = 0.0 + 0.0j
                temp_imode_just_diag = 0.0 + 0.0j
                for ik in range(Nkpoints):
                    for ind_cv1 in range(len(indexes_cv)):
                        # diagonal term (cv,cv)
                        ic1, iv1 = indexes_cv[ind_cv1]

                        temp = calc_Dkinect_matrix_elem(
                            Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic1, iv1, iv1)
                        # DKinect[imode, ik, ic1, iv1, ik, ic1, iv1] = temp

                        temp_imode_just_diag += temp  

                        # Now get offdiag
                        # don't get it repeated
                        for ind_cv2 in range(ind_cv1+1, len(indexes_cv)):

                            ic2, iv2 = indexes_cv[ind_cv2]

                            temp = calc_Dkinect_matrix_elem(
                                Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2)
                            # DKinect[imode, ik, ic1, iv1, ik, ic2, iv2] = temp
                            # DKinect[imode, ik, ic2, iv2, ik,
                            #         ic1, iv1] = np.conj(temp)

                            temp_imode_not_diag += 2 * np.real(temp)  
                            # temp + conj(temp)

                            # reporting
                            counter += 1
                            report_iterations(counter, total_iterations, interval_report, now_this_func)


                Sum_DKinect_diag[imode] = temp_imode_just_diag
                Sum_DKinect[imode] = temp_imode_not_diag + temp_imode_just_diag

    if report_RPA_data == True:
        print(f'RPA matrix elements written in file: RPA_matrix_elements.dat')

    return Sum_DKinect_diag, Sum_DKinect

def arg_lists_Dkinect(BSE_params, indexes_limited_BSE_sum):
    
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds_sum = BSE_params.Ncbnds_sum
    Nvbnds_sum = BSE_params.Nvbnds_sum
    
    args_list_just_offdiag = []
    
    if limit_BSE_sum == False:
        args_list_just_diag = [(ik, ic1, ic1, iv1, iv1) for ik in range(Nkpoints) for ic1 in range(Ncbnds_sum) for iv1 in range(Nvbnds_sum)]

        if just_RPA_diag == False:
            indexes_cv = [(icp, ivp) for ivp in range(Nvbnds_sum) for icp in range(Ncbnds_sum)]
            if use_hermicity_F == False:
                for cv1 in indexes_cv:
                    for cv2 in indexes_cv:
                        if cv1 != cv2:
                            ic1, iv1 = cv1
                            ic2, iv2 = cv2
                            for ik in range(Nkpoints): 
                                args_list_just_offdiag.append((ik, ic1, ic2, iv1, iv2))
            else:
                for i_cv1 in range(len(indexes_cv)):
                    for i_cv2 in range(i_cv1+1, len(indexes_cv)):
                        ic1, iv1 = indexes_cv[i_cv1]
                        ic2, iv2 = indexes_cv[i_cv2]
                        for ik in range(Nkpoints): 
                            args_list_just_offdiag.append((ik, ic1, ic2, iv1, iv2))

    else:
        args_list_just_diag = []
        
        # building diagonal terms
        for icvk1 in indexes_limited_BSE_sum:
            ik1, ic1, iv1 = icvk1
            args_list_just_diag.append((ik1, ic1, ic1, iv1, iv1))

        if just_RPA_diag == False:
            if use_hermicity_F == False:
                for icvk1 in indexes_limited_BSE_sum:
                    ik1, ic1, iv1 = icvk1
                    for icvk2 in indexes_limited_BSE_sum:
                        ik2, ic2, iv2 = icvk2
                        if ik1 == ik2:
                            args_list_just_offdiag.append((ik1, ic1, ic2, iv1, iv2))
            else:
                for icvk_index1 in range(len(indexes_limited_BSE_sum)):
                    ik1, ic1, iv1 = indexes_limited_BSE_sum[icvk_index1]
                    for icvk_index2 in range(icvk_index1+1, len(indexes_limited_BSE_sum)):  
                        ik2, ic2, iv2 = indexes_limited_BSE_sum[icvk_index2]
                        if ik1 == ik2:
                            args_list_just_offdiag.append((ik1, ic1, ic2, iv1, iv2))
                        
    return args_list_just_diag, args_list_just_offdiag


def calc_Dkinect_matrix_simplified(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, args_list, imode):

    result = 0.0 + 0.0j
    
    counter_now = 0
    total_iterations = len(args_list)
    when_function_started = datetime.now()
    
    for arg in args_list:
        ik, ic1, ic2, iv1, iv2 = arg
        result += calc_Dkinect_matrix_elem(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2)
        
        counter_now += 1
        report_iterations(counter_now, total_iterations, step_report, when_function_started)
    return result

def calc_Dkinect_matrix_parallel(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, args_list, imode):

    
    print('STEP 1')
    # Create a partial function with fixed arguments using functools.partial
    partial_func = partial(calc_Dkinect_matrix_elem, Akcv=Akcv, Bkcv=Bkcv, aux_cond_matrix=aux_cond_matrix, aux_val_matrix=aux_val_matrix, imode=imode)
    
    print('STEP 2')
    # Get the number of available processes
    num_processes = 1 # cpu_count()
    
    print('STEP 3')
    # Create a Pool object for parallel processing
    with Pool(processes=num_processes) as pool:
        # Use pool.map to apply the partial function to the args_list in parallel
        results = pool.map(partial_func, args_list)
    
    print('STEP 4')
    # Compute the sum of the results
    result = np.sum(np.array(results))
    return result


def calc_Dkinect_matrix_elem_parallel(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2):
    """Calculates excited state force matrix elements."""

    # calculate matrix element imode, ik, ic1, ic2, iv1, iv2
    temp_cond = aux_cond_matrix[imode, ik, ic1, ic2] * delta(iv1, iv2)
    temp_val = aux_val_matrix[imode, ik, iv1, iv2] * delta(ic1, ic2)

    tempA = Akcv[ik, ic1, iv1] * np.conj(Bkcv[ik, ic2, iv2])
    Dkin = tempA * (temp_cond - temp_val)

    return Dkin


def calculate_temp(args):
    imode, ik, ic1, ic2, iv1, iv2 = args
    return calc_Dkinect_matrix_elem_parallel(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2)


# def calc_Dkinect_matrix_parallel(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, MF_params, BSE_params):

#     print("\n\n     - Calculating RPA part")

#     Nmodes = MF_params.Nmodes
#     Nkpoints = BSE_params.Nkpoints_BSE
#     Ncbnds_sum = BSE_params.Ncbnds_sum
#     Nvbnds_sum = BSE_params.Nvbnds_sum

#     Sum_DKinect_diag = np.zeros((Nmodes), dtype=complex)
#     Sum_DKinect      = np.zeros((Nmodes), dtype=complex)

#     if just_RPA_diag == True:
        
#         print('Calculating just diag RPA force matrix elements - PARALLEL VERSION')
                
#         for imode in range(Nmodes):
#             temp_imode_just_diag = 0.0 + 0.0j

#             # Generate the list of arguments for the parallel execution
#             args_list = [(imode, ik, ic1, ic1, iv1, iv1) for ik in range(Nkpoints) for ic1 in range(Ncbnds_sum) for iv1 in range(Nvbnds_sum)]

#             # Parallel execution of calculate_temp function
#             temp_list = pool.map(calculate_temp, args_list)
#             temp_imode_just_diag = sum(temp_list)

#             Sum_DKinect_diag[imode] = temp_imode_just_diag            
            
#     else:

#         print('Calculating diag and offdiag RPA force matrix elements - PARALLEL VERSION')

#         if use_hermicity_F == False:

#             for imode in range(Nmodes):
                
#                 # Generate the list of arguments for the parallel execution
#                 args_list = [(imode, ik, ic1, ic1, iv1, iv1) for ik in range(Nkpoints) for ic1 in range(Ncbnds_sum) for iv1 in range(Nvbnds_sum)]
#                 args_list_include_offdiags = [(imode, ik, ic1, ic2, iv1, iv2) for ik in range(Nkpoints) for ic1 in range(Ncbnds_sum) for ic2 in range(Ncbnds_sum) for iv1 in range(Nvbnds_sum) for iv2 in range(Nvbnds_sum)]

#                 # TODO - exclude all elements from args_list_include_offdiags that are present in args_list
                
#                 # Parallel execution of calculate_temp function
#                 temp_list = pool.map(calculate_temp, args_list)
#                 temp_imode_just_diag = sum(temp_list)  
                
#                 # Parallel execution of calculate_temp function
#                 temp_list = pool.map(calculate_temp, args_list_include_offdiags)
#                 temp_imode_just_diag_and_off_diag = sum(temp_list) 

#                 Sum_DKinect_diag[imode] = temp_imode_just_diag
#                 Sum_DKinect[imode]      = temp_imode_just_diag_and_off_diag

#         # Creating a list of tuples with cond and val bands indexes
#         # [(0,0), (0,1), (0,2), ... (Ncbnds-1, 0), (Ncbnds-1, 1), ..., (Ncbnds-1, Nvbnds-1)]
#         # size of this list Nvbnds*Ncbnds

#         # New block - now using the fact that F_cvc'v' = conj(F_c'v'cv)
#         # Reduces the number of computed terms by about half

#         # Cannot use this if iexc != jexc 
#         # If use_hermicity_F == True and iexc != jexc
#         # then I just make use_hermicity_F == False

#         else:

#             print('Using "hermicity" of force matrix elements - PARALLEL VERSION')

#             indexes_cv = [(icp, ivp) for ivp in range(Nvbnds_sum) for icp in range(Ncbnds_sum)]

#             for imode in range(Nmodes):
                
#                 # building arg list
#                 args_list = [(imode, ik, ic1, ic1, iv1, iv1) for ik in range(Nkpoints) for ic1 in range(Ncbnds_sum) for iv1 in range(Nvbnds_sum)]
#                 args_list_include_offdiags = []
#                 for ik in range(Nkpoints):
#                     for ind_cv1 in range(len(indexes_cv)):
#                         ic1, iv1 = indexes_cv[ind_cv1]
#                         for ind_cv2 in range(ind_cv1+1, len(indexes_cv)):
#                             ic2, iv2 = indexes_cv[ind_cv2]
#                             args_list_include_offdiags.append((imode, ik, ic1, ic2, iv1, iv2))
                
#                 # TODO - exclude all elements from args_list_include_offdiags that are present in args_list
                
#                 # Parallel execution of calculate_temp function
#                 temp_list = pool.map(calculate_temp, args_list)
#                 temp_imode_just_diag = sum(temp_list)  
                
#                 # Parallel execution of calculate_temp function
#                 temp_list = pool.map(calculate_temp, args_list_include_offdiags)
#                 temp_imode_just_diag_and_off_diag = 2*np.real(sum(temp_list)) 

#                 Sum_DKinect_diag[imode] = temp_imode_just_diag
#                 Sum_DKinect[imode]      = temp_imode_just_diag_and_off_diag    

#     pool.close()
#     pool.join()
    
    # return Sum_DKinect_diag, Sum_DKinect





def calc_Dkinect_matrix_ver2_master(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, MF_params, BSE_params, KCV_list, run_parallel):

    ''' Updated version of calc_Dkinect_matrix, trying to be more organized'''
    
    print("\n\n     - Calculating RPA part")

    # getting variables
    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds_sum = BSE_params.Ncbnds_sum
    Nvbnds_sum = BSE_params.Nvbnds_sum

    Sum_DKinect_diag = np.zeros((Nmodes), dtype=complex)
    Sum_DKinect      = np.zeros((Nmodes), dtype=complex)
    
    # Just diag mat elements 
    for imode in range(Nmodes):
        Sum_DKinect_diag[imode] = calc_Dkinect_matrix_RPA_just_diag(imode, Nkpoints, Ncbnds_sum, Nvbnds_sum, Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, KCV_list)
    
    if just_RPA_diag == False:    
        
        print('Calculating diag and offdiag RPA force matrix elements')
        
        if use_hermicity_F == False:
            for imode in range(Nmodes):
                Sum_DKinect[imode] = Sum_DKinect_diag[imode] + calc_Dkinect_matrix_RPA_just_offdiag(imode, Nkpoints, Ncbnds_sum, Nvbnds_sum, Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, KCV_list)
            
    print(Sum_DKinect_diag, type(Sum_DKinect_diag))
    return Sum_DKinect_diag, Sum_DKinect

def calc_Dkinect_matrix_RPA_just_diag(imode, Nkpoints, Ncbnds_sum, Nvbnds_sum, Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, KCV_list):
    
    ''' Calculate Dkinect matrix for the case just_RPA == True'''
    
    KCV_array = np.array(KCV_list)
    ik_array = KCV_array[:, 0]
    ic_array = KCV_array[:, 1]
    iv_array = KCV_array[:, 2]
    
    calc_temp = np.vectorize(calc_Dkinect_matrix_elem, excluded=['Akcv', 'Bkcv', 'aux_cond_matrix', 'aux_val_matrix', 'imode'])

    temp_imode_just_diag = calc_temp(Akcv=Akcv, Bkcv=Bkcv, aux_cond_matrix=aux_cond_matrix, aux_val_matrix=aux_val_matrix, imode=imode, ik=ik_array, ic1=ic_array, ic2=ic_array, iv1=iv_array, iv2=iv_array)
                 
    return np.sum(temp_imode_just_diag)


def calc_Dkinect_matrix_RPA_just_offdiag(imode, Nkpoints, Ncbnds_sum, Nvbnds_sum, Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, KCV_list):
    
    ''' Calculate Dkinect matrix for the case just_RPA == True'''
    
    Total_ikcv = len(KCV_list)
    CV_list = KCV_list[:Ncbnds_sum*Nvbnds_sum]
    Total_icv = len(CV_list)
    
    # i am making Nkpoints * Total_icv * (Total_icv + 1) / 2 calculations
    temp_imode_just_diag = np.zeros(Nkpoints * Total_icv * (Total_icv + 1) / 2, dtype = complex)
    
    
    counter = -1
    for ik in range(Nkpoints):
        for icv in range(Total_icv):
            _, ic1, iv1 = CV_list[icv]
            for jcv in range(icv + 1, Total_icv):
                _, ic2, iv2 = CV_list[jcv]
                
                counter += 1
                temp_imode_just_diag[counter] = calc_Dkinect_matrix_elem(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2)
                      
    return np.sum(temp_imode_just_diag)






def calc_Dkinect_matrix_elem_wrapper(args):
    return calc_Dkinect_matrix_elem(*args)

def calc_Dkinect_matrix_just_RPA_para_ver(imode, Nkpoints, Ncbnds_sum, Nvbnds_sum, Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, KCV_list):
    Total_ikcv = len(KCV_list)
    args_list = [(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic, ic, iv, iv) for ik, ic, iv in KCV_list]
    with Pool() as pool:
        temp_imode_just_diag = pool.map(calc_Dkinect_matrix_elem_wrapper, args_list)
    return np.sum(temp_imode_just_diag)

# def calc_Dkinect_matrix_just_RPA_para_ver(imode, Nkpoints, Ncbnds_sum, Nvbnds_sum, Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, KCV_list):
    
#     ''' Calculate Dkinect matrix for the case just_RPA == True
#         - parallel version'''
    
#     Total_ikcv = len(KCV_list)
#     temp_imode_just_diag = np.zeros(Total_ikcv, dtype = complex)
    
#     def calc_elem(idx):
#         ik, ic, iv = KCV_list[idx]
#         return calc_Dkinect_matrix_elem(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic, ic, iv, iv)
    
#     with Pool() as pool:
#         result = pool.map(calc_elem, range(Total_ikcv))
    
#     return np.sum(result)


def step_report(total_iterations):
    
    return max(int(total_iterations / 20) - 1, 1)

def report_iterations(counter_now, total_iterations, step_report, when_function_started):
    
    ''' This function reports in iterarations and estimates how much time
    to finish some loop
    counter_now: current iteration
    total_iterations: total number of iterations
    step_report: number of iterations between each report 
    when_function_started: time when the function was started
    '''
    
    if counter_now % step_report == 0 or counter_now == 10 or counter_now == 10000:
        
        # how much time in seconds
        delta_T = (datetime.now() - when_function_started).total_seconds()
        # print(delta_T) 
        delta_T_remain = (total_iterations - counter_now) / counter_now * delta_T      
        
        print(f'{counter_now} of {total_iterations} calculated | {round(100*counter_now/total_iterations, 1)} % | elapsed {round(delta_T, 1)} s, remaining {round(delta_T_remain, 1)} s')