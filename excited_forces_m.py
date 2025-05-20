

# TODO -> splits this module for the different functions
#          -> reads GW/BSE data
#          -> reads DFPT data
#          -> calculates stuff
#          -> etc

from modules_to_import import *
from excited_forces_config import *
from bgw_interface_m import *
from qe_interface_m import *
from excited_forces_classes import *
    
def report_expected_energies(Akcv, Omega, Eqp_cond, Eqp_val):

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
    
def get_exciton_coeffs(iexc, jexc):
    if config["read_Acvk_pos"] == False:
        Akcv = get_exciton_info(config["exciton_file"], iexc)
        if iexc == jexc:
            Bkcv = Akcv
        else:
            Bkcv = get_exciton_info(config["exciton_file"], jexc)    
    else:
        Akcv = get_exciton_info_alternative(config["Acvk_directory"], iexc, Nkpoints_BSE, Ncbnds, Nvbnds)
        if iexc == jexc:
            Bkcv = Akcv
        else:
            Bkcv = get_exciton_info_alternative(config["Acvk_directory"], jexc, Nkpoints_BSE, Ncbnds, Nvbnds)
    return Akcv, Bkcv


def report_expected_energies_master(iexc, jexc, Eqp_cond, Eqp_val, Akcv, OmegaA, Bkcv, OmegaB):         
    # Reporting expected energies
    if iexc != jexc:
        print(f'Exciton {iexc}')
        report_expected_energies(Akcv, OmegaA, Eqp_cond, Eqp_val)
        print(f'Exciton {jexc}')
        report_expected_energies(Bkcv, OmegaB, Eqp_cond, Eqp_val)
    else:
        print(f'Exciton {iexc}')
        report_expected_energies(Akcv, OmegaA, Eqp_cond, Eqp_val)

    
    
def generate_indexes_limited_BSE_sum():
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
        
    return indexes_limited_BSE_sum
    
def summarize_Acvk(Akcv, Kpoints_BSE, indexes_limited_BSE_sum):
    if len(indexes_limited_BSE_sum) > 0:
        top_indexes = indexes_limited_BSE_sum
    else:
        top_indexes = top_n_indexes(np.abs(Akcv), 10)
        
    print('###############################################')
    print('Showing most relevant coeffs for this exciton')
    print('kx        ky        kz        ic   iv   abs(Acvk)^2  partial_sum(abs(Acvk)^2)')
    partial_sum = 0
    for index_Acvk in top_indexes:
        ik, ic, iv = index_Acvk
        A = Akcv[index_Acvk]
        partial_sum += abs(A)**2
        kx, ky, kz = Kpoints_BSE[ik, 0], Kpoints_BSE[ik, 1], Kpoints_BSE[ik, 2]
        print(f'{kx:8.4f}  {ky:8.4f}  {kz:8.4f}  {ic+1:<3} {iv+1:<3} {abs(A)**2:10.4f}   {partial_sum:10.4f}')    
    print('###############################################') 
    
    return top_indexes 

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
    
def get_BSE_MF_params():

    global MF_params, BSE_params, Nmodes
    global Nat, atomic_pos, cell_vecs, cell_vol, alat
    global Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval
    global Nvbnds_sum, Ncbnds_sum
    global Nvbnds_coarse, Ncbnds_coarse, Nkpoints_coarse
    global rec_cell_vecs, Nmodes

    if config["read_Acvk_pos"] == False:
        Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs, NQ, Qshift = get_params_from_eigenvecs_file(config["exciton_file"])
    else:
        Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs = get_params_from_alternative_file('params')
        print("Not reading eigevectors.h5 produced from absorption step. Assuming that this calculation has no Q shift.")
        NQ = 1 
        Qshift = np.zeros((3), dtype=float)
    
    Nmodes = 3 * Nat

    if 0 < config["ncbnds_sum"] < Ncbnds:
        print('*********************************')
        print('Instead of using all cond bands from the BSE hamiltonian')
        print(f'I will use {ncbnds_sum} cond bands (variable ncbnds_sum)')
        print('*********************************')
        Ncbnds_sum = ncbnds_sum
    else:
        Ncbnds_sum = Ncbnds

    if 0 < config["nvbnds_sum"] < Nvbnds:
        print('*********************************')
        print('Instead of using all val bands from the BSE hamiltonian')
        print(f'I will use {config["nvbnds_sum"]} val bands (variable nvbnds_sum)')
        print('*********************************')
        Nvbnds_sum = config["nvbnds_sum"] #nvbnds_sum
    else:
        Nvbnds_sum = Nvbnds

    if config["elph_fine_a_la_bgw"] == True:
        print('I will perform elph interpolation "a la BerkeleyGW"')
        print('Check the absorption.inp file to see how many bands were used in both coarse and fine grids.')
        print('From the forces.inp file, I got the following parameters: ')
        print(f'    ncond_coarse    = {config["ncbands_co"]}')
        print(f'    nval_coarse     = {config["nvbands_co"]}')
        print(f'    nkpoints_coarse = {config["nkpnts_co"]}')
        print('Be sure that all those bands are included in the DFPT calculation!')
        print('If not, the missing elph coefficients will be considered to be equal 0.')

    Ncbnds_coarse = config["ncbands_co"]
    Nvbnds_coarse = config["nvbands_co"]
    Nkpoints_coarse = config["nkpnts_co"]


    MF_params = Parameters_MF(Nat, atomic_pos, cell_vecs, cell_vol, alat)
    BSE_params = Parameters_BSE(Nkpoints_BSE, Kpoints_BSE, Ncbnds, Nvbnds, Nval, Ncbnds_sum, Nvbnds_sum, Ncbnds_coarse, Nvbnds_coarse, Nkpoints_coarse, rec_cell_vecs)
    
    return Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs, BSE_params, MF_params, NQ, Qshift

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
    
    



def top_n_indexes(array, N):
    # Flatten the array
    flat_array = array.flatten()
    
    # Get the indexes of the top N values in the flattened array
    flat_indexes = np.argpartition(flat_array, -N)[-N:]
    
    # Sort these indexes by the values they point to, in descending order
    sorted_indexes = flat_indexes[np.argsort(-flat_array[flat_indexes])]
    
    # Convert the 1D indexes back to 3D indexes
    top_indexes = np.unravel_index(sorted_indexes, array.shape)
    
    # Combine the indexes into a list of tuples
    top_indexes = list(zip(*top_indexes))
    
    return top_indexes

def elph_renormalization_matrix(Eqp_val, Eqp_cond, Edft_val, Edft_cond, BSE_params, ikBSE_to_ikDFPT):
    """ Calculates auxiliar matrix elements to be used later in the forces matrix elements.
    Returns elph_renormalization_matrix_cond, elph_renormalization_matrix_val where
    elph_renormalization_matrix_cond[imode, ik, ic1, ic2] = deltaEqp / deltaEdft if deltaEdft <= TOL_DEG or 1
    elph_renormalization_matrix_val[imode, ik, iv1, iv2]  = deltaEqp / deltaEdft if deltaEdft <= TOL_DEG or 1 
    If deltaEdft == 0, then the matrix elements are just the elph coefficients"""

    now_this_func = datetime.now()
    
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds_sum = BSE_params.Ncbnds_sum
    Nvbnds_sum = BSE_params.Nvbnds_sum

    Shape_cond = (Nkpoints, Ncbnds_sum, Ncbnds_sum)
    renorm_matrix_cond = np.ones(Shape_cond, dtype=np.complex64)

    Shape_val = (Nkpoints, Nvbnds_sum, Nvbnds_sum)
    renorm_matrix_val = np.ones(Shape_val, dtype=np.complex64)
    
    total_iterations = Nkpoints
    interval_report = step_report(total_iterations)
    counter = 0
    for ik in range(Nkpoints):

        ik_dfpt = ikBSE_to_ikDFPT[ik]

        if ik_dfpt != -1:
            # remember that order of k points in DFPT file
            # is not always equal to the order in the eigenvecs file
            # -1 means that the code did not find the corresponce
            # between the 2 kgrids

            for ic1 in range(Ncbnds_sum):
                for ic2 in range(Ncbnds_sum):
                    deltaEqp = Eqp_cond[ik, ic1] - Eqp_cond[ik, ic2]
                    deltaEdft = Edft_cond[ik, ic1] - Edft_cond[ik, ic2]
                    if abs(deltaEdft) > TOL_DEG:
                        renorm_matrix_cond[ik, ic1, ic2] = deltaEqp / deltaEdft
                        # renorm_matrix_cond[ik, ic2, ic1] = deltaEqp / deltaEdft
                        
            for iv1 in range(Nvbnds_sum):
                for iv2 in range(Nvbnds_sum):
                    deltaEqp = Eqp_val[ik, iv1] - Eqp_val[ik, iv2]
                    deltaEdft = Edft_val[ik, iv1] - Edft_val[ik, iv2]
                    if abs(deltaEdft) > TOL_DEG:
                        renorm_matrix_val[ik, iv1, iv2] = deltaEqp / deltaEdft
                        # renorm_matrix_val[ik, iv2, iv1] = deltaEqp / deltaEdft
                            
        else:
            print(f'Kpoint {ik} not found. Skipping the calculation for this k point')
            
        counter += 1
        report_iterations(counter, total_iterations, interval_report, now_this_func)

    return renorm_matrix_cond, renorm_matrix_val


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

    if config["write_dK_mat"] == True:
        arq_dK_report.close()
    
    return Sum_DKernel
    

def renormalize_elph_considering_kpt_order(elph_cond, elph_val, Eqp_val, Eqp_cond, Edft_val, Edft_cond, MF_params, BSE_params, ikBSE_to_ikDFPT):
    """ Calculates auxiliar matrix elements to be used later in the forces matrix elements.
    Returns aux_cond_matrix, aux_val_matrix and
    aux_cond_matrix[imode, ik, ic1, ic2] = elph_cond[imode, ik, ic1, ic2] * deltaEqp / deltaEdft (if ic1 != ic2)
    aux_val_matrix[imode, ik, iv1, iv2]  = elph_val[imode, ik, iv1, iv2]  * deltaEqp / deltaEdft (if iv1 != iv2)
    If deltaEdft == 0, then the matrix elements are just the elph coefficients"""

    now_this_func = datetime.now()
    
    Nmodes = MF_params.Nmodes
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds_sum = BSE_params.Ncbnds_sum
    Nvbnds_sum = BSE_params.Nvbnds_sum

    Shape_cond = (Nmodes, Nkpoints, Ncbnds_sum, Ncbnds_sum)
    aux_cond_matrix = np.zeros(Shape_cond, dtype=np.complex64)

    Shape_val = (Nmodes, Nkpoints, Nvbnds_sum, Nvbnds_sum)
    aux_val_matrix = np.zeros(Shape_val, dtype=np.complex64)
    
    total_iterations = Nmodes*Nkpoints
    interval_report = step_report(total_iterations)
    counter = 0

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

                        if config["no_renorm_elph"] == True:
                            aux_cond_matrix[imode, ik, ic1, ic2] = elph

                        else:
                            deltaEqp = Eqp_cond[ik, ic1] - Eqp_cond[ik, ic2]
                            deltaEdft = Edft_cond[ik, ic1] - Edft_cond[ik, ic2]

                            if abs(deltaEdft) <= TOL_DEG:
                                aux_cond_matrix[imode, ik, ic1, ic2] = elph
                            else:
                                aux_cond_matrix[imode, ik, ic1, ic2] = elph * deltaEqp / deltaEdft
                            

                for iv1 in range(Nvbnds_sum):
                    for iv2 in range(Nvbnds_sum):

                        elph = elph_val[imode, ik_dfpt, iv1, iv2]

                        if config["no_renorm_elph"] == True:
                            aux_val_matrix[imode, ik, iv1, iv2] = elph

                        else:
                            deltaEqp = Eqp_val[ik, iv1] - Eqp_val[ik, iv2]
                            deltaEdft = Edft_val[ik, iv1] - Edft_val[ik, iv2]

                            if abs(deltaEdft) <= TOL_DEG:
                                aux_val_matrix[imode, ik, iv1, iv2] = elph
                            else:
                                aux_val_matrix[imode, ik, iv1, iv2] = elph * deltaEqp / deltaEdft
                                
            else:
                print(f'Kpoint {ik} not found. Skipping the calculation for this k point')
                
            counter += 1
            report_iterations(counter, total_iterations, interval_report, now_this_func)

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
    Nc = BSE_params.Ncbnds
    Nv = BSE_params.Nvbnds
    
    args_list_just_offdiag = []
    
    if len(indexes_limited_BSE_sum) == 0:
        print("Not limiting sum over KCV indexes")
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
        print("Limiting sum over KCV indexes")
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
    
    Ntransitions = Nkpoints*Nc*Nv
    if use_hermicity_F == False:
        Ntransitions_offdiag = Nkpoints*(Nc*Nv)**2 - Ntransitions
    else:
        Ntransitions_offdiag = int((Nkpoints*(Nc*Nv)**2 - Ntransitions) / 2)
    print("Original number of diagonal matrix elements ", Ntransitions)
    print("Original number of off-diagonal matrix elements ", Ntransitions_offdiag)
    print("Number of diagonal matrix elements (kcv -> kcv) to be calculated = ", len(args_list_just_diag))
    print("Number of off-diagonal matrix elements (kcv -> kc'v') to be calculated = ", len(args_list_just_offdiag))
    return args_list_just_diag, args_list_just_offdiag


def calc_Dkinect_matrix_simplified(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, args_list, imode):

    result = 0.0 + 0.0j
    
    counter_now = 0
    total_iterations = len(args_list)
    when_function_started = datetime.now()
    step_report_here = step_report(total_iterations)
    
    for arg in args_list:
        ik, ic1, ic2, iv1, iv2 = arg
        result += calc_Dkinect_matrix_elem(Akcv, Bkcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2)
        
        counter_now += 1
        report_iterations(counter_now, total_iterations, step_report_here, when_function_started)
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


def step_report(total_iterations):
    
    return max(int(total_iterations / 10) - 1, 1)
 
def report_iterations(counter_now, total_iterations, step_report, when_function_started):
    '''
    This function reports in iterations and estimates how much time
    to finish some loop.
    
    counter_now: current iteration
    total_iterations: total number of iterations
    step_report: number of iterations between each report 
    when_function_started: time when the function was started
    '''
    
    if counter_now % step_report == 0 or counter_now == 10 or counter_now == 10000:
        # Calculate elapsed time in seconds
        delta_T = (datetime.now() - when_function_started).total_seconds()
        
        # Estimate remaining time
        delta_T_remain = (total_iterations - counter_now) / counter_now * delta_T
        
        # Format and print the report
        print(f'Transitions {counter_now:8} of {total_iterations:8} calculated | '
              f'{round(100 * counter_now / total_iterations, 1):5.1f} % | '
              f'elapsed {round(delta_T, 1):10.1f} s, remaining {round(delta_T_remain, 1):10.1f} s')
        
        
def apply_Qshift_on_valence_states(Qshift, Gv, Kpoints_in_elph_file_frac):
    if np.linalg.norm(Qshift) > 0.0:
        print(f"Applying Q shift to valence states")

        # shape Qshift is (3,)
        # shape Kpoints_in_elph_file_frac is (Nkpoints_DFPT, 3)
        Kpoints_shifted = (Kpoints_in_elph_file_frac + Qshift) % 1.0  # %1.0 is to put in the first BZ
        
        mapping = []
        for kshifted in Kpoints_shifted:
            distances = np.linalg.norm(Kpoints_in_elph_file_frac - kshifted, axis = 1)
            min_index = np.argmin(distances)
            if distances[min_index] >  1e-4:
                print(f"WARNING! The Q-shifted k point {kshifted} is not close to any k point in the DFPT calculation. The closest one is {Kpoints_in_elph_file_frac[min_index]} with distance {distances[min_index]}")
            mapping.append(min_index)
        
        Gv_new = Gv[:, mapping, :, :, :, :]
        Gv = Gv_new
        print(f"Done applying Q shift to valence states")
    else:
        print(f"NOT applying Q shift to valence states")

    return Gv
