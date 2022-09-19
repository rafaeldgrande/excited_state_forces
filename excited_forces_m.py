

# TODO -> splits this module for the different functions
#          -> reads GW/BSE data
#          -> reads DFPT data
#          -> calculates stuff
#          -> etc


import numpy as np
import xml.etree.ElementTree as ET
import h5py

from excited_forces_config import *


def calc_DKernel_mat_elem(indexes, Kernel, EDFT, EQP, ELPH, MF_params, BSE_params):
    """Calculates derivatives of kernel matrix elements"""

    ik1, ik2, iv1, iv2, ic1, ic2, imode = indexes

    elph_cond, elph_val = ELPH
    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds = BSE_params.Ncbnds
    Nvbnds = BSE_params.Nvbnds
    Edft_val, Edft_cond = EDFT

    if calc_IBL_way == True:
        Eqp_val, Eqp_cond = EQP
        DKelement_IBL = 0 + 0.0*1.0j

    DKelement = 0 + 0.0*1.0j

    # ik2, ik1, ic2, ic1, iv2, iv1

    for ivp in range(Nvbnds):

        DeltaEdft = Edft_val[ik1, iv1] - Edft_val[ik1, ivp]
        if calc_IBL_way == True:
            DeltaEqp = Eqp_val[ik1, iv1] - Eqp_val[ik1, ivp]

        indexes_K = ik2, ik1, ic2, ic1, iv2, ivp
        indexes_g = imode, ik1, ivp, iv1

        if abs(DeltaEdft) > TOL_DEG:
            DKelement += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEdft
            if calc_IBL_way == True:  # assuming that gw levels have the same degeneracy of dft levels
                DKelement_IBL += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEqp

        DeltaEdft = Edft_val[ik2, iv2] - Edft_val[ik2, ivp]
        if calc_IBL_way == True:
            DeltaEqp = Eqp_val[ik2, iv2] - Eqp_val[ik2, ivp]

        indexes_K = ik2, ik1, ic2, ic1, ivp, iv1
        indexes_g = imode, ik2, iv2, ivp

        if abs(DeltaEdft) > TOL_DEG:
            DKelement += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEdft
            if calc_IBL_way == True:
                DKelement_IBL += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEqp

    for icp in range(Ncbnds):

        DeltaEdft = Edft_cond[ik1, ic1] - Edft_cond[ik1, icp]
        if calc_IBL_way == True:
            DeltaEqp = Eqp_cond[ik1, ic1] - Eqp_cond[ik1, icp]

        indexes_K = ik2, ik1, ic2, icp, iv2, iv1
        indexes_g = imode, ik1, icp, ic1

        if abs(DeltaEdft) > TOL_DEG:
            DKelement += Kernel[indexes_K]*elph_cond[indexes_g]/DeltaEdft
            if calc_IBL_way == True:
                DKelement_IBL += Kernel[indexes_K] * \
                    elph_cond[indexes_g]/DeltaEqp

        DeltaEdft = Edft_cond[ik2, ic2] - Edft_cond[ik2, icp]
        if calc_IBL_way == True:
            DeltaEqp = Eqp_cond[ik2, ic2] - Eqp_cond[ik2, icp]

        indexes_K = ik2, ik1, icp, ic1, iv2, iv1
        indexes_g = imode, ik2, ic2, icp

        if abs(DeltaEdft) > TOL_DEG:
            DKelement += Kernel[indexes_K]*elph_cond[indexes_g]/DeltaEdft
            if calc_IBL_way == True:
                DKelement_IBL += Kernel[indexes_K] * \
                    elph_cond[indexes_g]/DeltaEqp

    if calc_IBL_way is True:
        return DKelement, DKelement_IBL
    else:
        return DKelement


def calc_deriv_Kernel(KernelMat, EDFT, EQP, ELPH, Akcv, Bkcv, MF_params, BSE_params):

    print("\n\n    - Calculating Kernel part")

    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds = BSE_params.Ncbnds
    Nvbnds = BSE_params.Nvbnds

    Shape_kernel = (Nmodes, Nkpoints, Ncbnds, Nvbnds, Nkpoints, Ncbnds, Nvbnds)
    DKernel = np.zeros(Shape_kernel, dtype=np.complex64)
    DKernel_IBL = np.zeros(Shape_kernel, dtype=np.complex64)

    total_ite = Nmodes*(Nkpoints*Ncbnds*Nvbnds)**2
    interval_report = max(int(np.sqrt(total_ite)), int(total_ite/20))
    i_term = 0
    print(f'        Total terms to be calculated : {total_ite}')

    for imode in range(Nmodes):

        for ik1 in range(Nkpoints):
            for ic1 in range(Ncbnds):
                for iv1 in range(Nvbnds):

                    A_bra = np.conj(Akcv[ik1, ic1, iv1])

                    for ik2 in range(Nkpoints):
                        for ic2 in range(Ncbnds):
                            for iv2 in range(Nvbnds):

                                A_ket = Bkcv[ik2, ic2, iv2]

                                indexes = ik1, ik2, iv1, iv2, ic1, ic2, imode
                                dK = calc_DKernel_mat_elem(
                                    indexes, KernelMat, EDFT, EQP, ELPH, MF_params, BSE_params)

                                if calc_IBL_way == False:
                                    DKernel[imode, ik1, ic1, iv1, ik2,
                                            ic2, iv2] = A_bra * dK * A_ket
                                else:
                                    DKernel[imode, ik1, ic1, iv1, ik2,
                                            ic2, iv2] = A_bra * dK[0] * A_ket
                                    DKernel_IBL[imode, ik1, ic1, iv1, ik2,
                                                ic2, iv2] = A_bra * dK[1] * A_ket

                                i_term += 1
                                if i_term % interval_report == 0:
                                    print(
                                        f'        {i_term} of {total_ite} calculated --------- {round(100*i_term/total_ite,1)} %')

    if calc_IBL_way == False:
        return DKernel
    else:
        return DKernel, DKernel_IBL


def aux_matrix_elem(elph_cond, elph_val, Eqp_val, Eqp_cond, Edft_val, Edft_cond, MF_params, BSE_params, ikBSE_to_ikDFPT):
    """ Calculates auxiliar matrix elements to be used later in the forces matrix elements.
    Returns aux_cond_matrix, aux_val_matrix and
    aux_cond_matrix[imode, ik, ic1, ic2] = elph_cond[imode, ik, ic1, ic2] * deltaEqp / deltaEdft (if ic1 != ic2)
    aux_val_matrix[imode, ik, iv1, iv2]  = elph_val[imode, ik, iv1, iv2]  * deltaEqp / deltaEdft (if iv1 != iv2)
    If ic1 == ic2 (iv1 == iv2), then the matrix elements are just the elph coefficients"""

    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds = BSE_params.Ncbnds
    Nvbnds = BSE_params.Nvbnds

    Shape_cond = (Nmodes, Nkpoints, Ncbnds, Ncbnds)
    aux_cond_matrix = np.zeros(Shape_cond, dtype=np.complex64)

    Shape_val = (Nmodes, Nkpoints, Nvbnds, Nvbnds)
    aux_val_matrix = np.zeros(Shape_val, dtype=np.complex64)

    for imode in range(Nmodes):
        for ik in range(Nkpoints):

            ik_dfpt = ikBSE_to_ikDFPT[ik]

            if ik_dfpt != -1:
                # remember that order of k points in DFPT file
                # is not always equal to the order in the eigenvecs file
                # -1 means that the code did not find the corresponce
                # between the 2 kgrids

                for ic1 in range(Ncbnds):
                    for ic2 in range(Ncbnds):

                        elph = elph_cond[imode, ik_dfpt, ic1, ic2]

                        if ic1 == ic2:
                            aux_cond_matrix[imode, ik, ic1, ic2] = elph

                        elif abs(Edft_cond[ik, ic1] - Edft_cond[ik, ic2]) > TOL_DEG:
                            deltaEqp = Eqp_cond[ik, ic1] - Eqp_cond[ik, ic2]
                            deltaEdft = Edft_cond[ik, ic1] - Edft_cond[ik, ic2]
                            aux_cond_matrix[imode, ik, ic1,
                                            ic2] = elph * deltaEqp / deltaEdft

                for iv1 in range(Nvbnds):
                    for iv2 in range(Nvbnds):

                        elph = elph_val[imode, ik_dfpt, iv1, iv2]

                        if iv1 == iv2:
                            aux_val_matrix[imode, ik, iv1, iv2] = elph

                        elif abs(Edft_val[ik, iv1] - Edft_val[ik, iv2]) > TOL_DEG:
                            deltaEqp = Eqp_val[ik, iv1] - Eqp_val[ik, iv2]
                            deltaEdft = Edft_val[ik, iv1] - Edft_val[ik, iv2]
                            aux_val_matrix[imode, ik, iv1,
                                           iv2] = elph * deltaEqp / deltaEdft

    return aux_cond_matrix, aux_val_matrix


def delta(i, j):
    """Dirac delta - TODO: check if python has a builtin function that does that."""
    if i == j:
        return 1.0
    else:
        return 0.0


def dirac_delta_Edft(i, j, Edft, TOL_DEG):
    if abs(Edft[0, i] - Edft[0, j]) > TOL_DEG:
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

    print("\n\n     - Calculating RPA part")

    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds = BSE_params.Ncbnds
    Nvbnds = BSE_params.Nvbnds

    Shape = (Nmodes, Nkpoints, Ncbnds, Nvbnds, Nkpoints, Ncbnds, Nvbnds)
    DKinect = np.zeros(Shape, dtype=np.complex64)

    if report_RPA_data == True:
        arq_RPA_data = open('RPA_matrix_elements.dat', 'w')
        arq_RPA_data.write(
            '# mode ik ic1 ic2 iv1 iv2 F conj(Akc1v1)*Akc2v2 auxMatcond(c1,c2) auxMatval(v1,v2)\n')
        arq_RPA_data.close()

    if just_RPA_diag == True:

        # reporting
        total_ite = Nmodes*Nkpoints*Ncbnds*Nvbnds
        interval_report = max(int(np.sqrt(total_ite)), int(total_ite/20))
        i_term = 0

        print('Calculating just diag RPA force matrix elements')
        print(f'Total terms to be calculated : {total_ite}')

        for imode in range(Nmodes):
            for ik in range(Nkpoints):
                for ic1 in range(Ncbnds):
                    for iv1 in range(Nvbnds):
                        temp = calc_Dkinect_matrix_elem(
                            Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic1, iv1, iv1)
                        DKinect[imode, ik, ic1, iv1, ik, ic1, iv1] = temp

                        # reporting
                        i_term += 1
                        if i_term % interval_report == 0:
                            print(
                                f'{i_term} of {total_ite} calculated --------- {round(100*i_term/total_ite,1)} %')

    else:

        print('Calculating diag and offdiag RPA force matrix elements')

        if use_hermicity_F == False:

            # reporting
            total_ite = Nmodes*Nkpoints*(Ncbnds*Nvbnds)**2
            interval_report = max(int(np.sqrt(total_ite)), int(total_ite/20))
            i_term = 0
            print(f'Total terms to be calculated : {total_ite}')

            for imode in range(Nmodes):
                for ik in range(Nkpoints):
                    for ic1 in range(Ncbnds):
                        for iv1 in range(Nvbnds):
                            for ic2 in range(Ncbnds):
                                for iv2 in range(Nvbnds):
                                    temp = calc_Dkinect_matrix_elem(
                                        Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2)
                                    DKinect[imode, ik, ic1, iv1,
                                            ik, ic2, iv2] = temp

                                    # reporting
                                    i_term += 1
                                    if i_term % interval_report == 0:
                                        print(
                                            f'{i_term} of {total_ite} calculated --------- {round(100*i_term/total_ite,1)} %')

        # Creating a list of tuples with cond and val bands indexes
        # [(0,0), (0,1), (0,2), ... (Ncbnds-1, 0), (Ncbnds-1, 1), ..., (Ncbnds-1, Nvbnds-1)]
        # size of this list Nvbnds*Ncbnds

        # New block - now using the fact that F_cvc'v' = conj(F_c'v'cv)
        # Reduces the number of computed terms by about half

        else:

            print('Using "hermicity" of force matrix elements')

            indexes_cv = [(icp, ivp) for ivp in range(Nvbnds)
                          for icp in range(Ncbnds)]

            # reporting
            total_ite = int(Nmodes*Nkpoints*len(indexes_cv)
                            * (len(indexes_cv)-1) / 2)
            interval_report = max(int(np.sqrt(total_ite)), int(total_ite/20))
            i_term = 0
            print(f'Total terms to be calculated : {total_ite}')

            for imode in range(Nmodes):
                for ik in range(Nkpoints):
                    for ind_cv1 in range(len(indexes_cv)):
                        # diagonal term (cv,cv)
                        ic1, iv1 = indexes_cv[ind_cv1]

                        temp = calc_Dkinect_matrix_elem(
                            Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic1, iv1, iv1)
                        DKinect[imode, ik, ic1, iv1, ik, ic1, iv1] = temp

                        # Now get offdiag
                        # don't get it repeated
                        for ind_cv2 in range(ind_cv1+1, len(indexes_cv)):

                            ic2, iv2 = indexes_cv[ind_cv2]

                            temp = calc_Dkinect_matrix_elem(
                                Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2)
                            DKinect[imode, ik, ic1, iv1, ik, ic2, iv2] = temp
                            DKinect[imode, ik, ic2, iv2, ik,
                                    ic1, iv1] = np.conj(temp)

                            # reporting
                            i_term += 1
                            if i_term % interval_report == 0:
                                print(
                                    f'       {i_term} of {int(total_ite)} calculated --------- {round(100*i_term/total_ite,1)} %')

    if report_RPA_data == True:
        print(f'RPA matrix elements written in file: RPA_matrix_elements.dat')

    return DKinect
