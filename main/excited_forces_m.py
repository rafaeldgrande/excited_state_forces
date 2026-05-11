

from modules_to_import import *
from excited_forces_config import *
from bgw_interface_m import *
from excited_forces_classes import *

def read_exciton_pairs(config):
    """
    Reads exciton pairs from 'exciton_pairs.dat' if config['read_exciton_pairs_file'] is True.
    Updates config['exciton_pairs'] accordingly.
    """
    if config.get('read_exciton_pairs_file', False):
        try:
            with open('exciton_pairs.dat', 'r') as arq:
                config['exciton_pairs'] = []
                for line in arq:
                    linha = line.split()
                    if len(linha) == 1:
                        config['exciton_pairs'].append((int(linha[0]), int(linha[0])))
                    elif len(linha) == 2:
                        config['exciton_pairs'].append((int(linha[0]), int(linha[1])))
            print("Reading exciton pairs from file exciton_pairs.dat. Ignoring iexc and jexc values from forces.inp file")
        except FileNotFoundError:
            print("Error: File 'exciton_pairs.dat' not found. Using default exciton pairs.")
            config['exciton_pairs'] = [(config['iexc'], config['jexc'])]
    else:
        config['exciton_pairs'] = [(config['iexc'], config['jexc'])]


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
        diff = kpoint - K_list[index]
        diff = diff - np.round(diff)   # fold to [-0.5, 0.5) to handle BZ translations
        if np.linalg.norm(diff) <= TOL_DEG:
            index_in_matrix = index
    return index_in_matrix


def get_BSE_MF_params():

    global MF_params, BSE_params, Nmodes
    global Nat, atomic_pos, cell_vecs, cell_vol, alat
    global Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval
    global Nvbnds_sum, Ncbnds_sum
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
        Nvbnds_sum = config["nvbnds_sum"]
    else:
        Nvbnds_sum = Nvbnds

    MF_params = Parameters_MF(Nat, atomic_pos, cell_vecs, cell_vol, alat)
    BSE_params = Parameters_BSE(Nkpoints_BSE, Kpoints_BSE, Ncbnds, Nvbnds, Nval, Ncbnds_sum, Nvbnds_sum, rec_cell_vecs)

    return Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs, BSE_params, MF_params, NQ, Qshift


def calc_Dkernel_new(kernel_matrix, elph_cond, elph_val, Econd, Eval, vectorized=True):

    nmodes = elph_cond.shape[0]
    nk, nc = Econd.shape
    _, nv = Eval.shape

    DKernel_dr_imode = np.zeros((nmodes, nk, nc, nv, nk, nc, nv), dtype=complex)

    for imode in range(nmodes):
        if vectorized:
            DKernel_dr_imode[imode] = calc_Dkernel_vectorized(kernel_matrix, elph_cond, elph_val, Econd, Eval, imode)
        else:
            DKernel_dr_imode[imode] = calc_Dkernel_single_mode_not_vectorized(kernel_matrix, elph_cond, elph_val, Econd, Eval, imode)
    return DKernel_dr_imode


def calc_Dkernel_single_mode_not_vectorized(kernel_matrix, elph_cond, elph_val, Econd, Eval, imode):
    """Calculates derivative of kernel matrix element for a single mode
    return the matrix d/dr_imode <kcv | K^{eh} | k'c'v'>
    shape nk, nc, nv, nk, nc, nv

    Units: kernel matrix elements are in eV, elph_cond and elph_val are in ry/bohr,
    Econd and Eval are in eV, so DKernel_dr_imode is in ry/bohr
    """

    nk, nc = Econd.shape
    _, nv = Eval.shape
    DKernel_dr_imode = np.zeros((nk, nc, nv, nk, nc, nv), dtype=complex)

    for ik1 in range(nk):
        for ik2 in range(nk):
            for iv1 in range(nv):
                for iv2 in range(nv):
                    for ic1 in range(nc):
                        for ic2 in range(nc):

                            temp_sum = 0.0 + 0.0j

                            for icp in range(nc):
                                if abs(Econd[ik1, ic1] - Econd[ik1, icp]) > TOL_DEG:
                                    temp_sum += kernel_matrix[ik1, icp, iv1, ik2, ic2, iv2] * elph_cond[imode, ik1, ic1, icp] / (Econd[ik1, ic1] - Econd[ik1, icp])

                                if abs(Econd[ik1, ic2] - Econd[ik1, icp]) > TOL_DEG:
                                    temp_sum += kernel_matrix[ik1, ic1, iv1, ik2, icp, iv2] * elph_cond[imode, ik2, icp, ic2] / (Econd[ik2, ic2] - Econd[ik2, icp])

                            for ivp in range(nv):
                                if abs(Eval[ik1, iv1] - Eval[ik1, ivp]) > TOL_DEG:
                                    temp_sum += kernel_matrix[ik1, ic1, ivp, ik2, ic2, iv2] * elph_val[imode, ik1, iv1, ivp] / (Eval[ik1, iv1] - Eval[ik1, ivp])

                                if abs(Eval[ik1, iv2] - Eval[ik1, ivp]) > TOL_DEG:
                                    temp_sum += kernel_matrix[ik1, ic1, iv1, ik2, ic2, ivp] * elph_val[imode, ik2, ivp, iv2] / (Eval[ik2, iv2] - Eval[ik2, ivp])

                            DKernel_dr_imode[ik1, ic1, iv1, ik2, ic2, iv2] = temp_sum
    return DKernel_dr_imode


def one_over_E_diff(Ebands):
    nk, nb = Ebands.shape
    E_diff_mask = np.zeros((nk, nb, nb), dtype=float)

    for ik in range(nk):
        for ib1 in range(nb):
            for ib2 in range(nb):
                E_diff = Ebands[ik, ib1] - Ebands[ik, ib2]
                if abs(E_diff) > TOL_DEG:
                    E_diff_mask[ik, ib1, ib2] = 1.0 / E_diff
                else:
                    E_diff_mask[ik, ib1, ib2] = 0.0
    return E_diff_mask


def calc_Dkernel_vectorized(kernel_matrix, elph_cond, elph_val, Econd, Eval, imode):
    nk, nc = Econd.shape
    _, nv = Eval.shape
    DKernel_dr_imode = np.zeros((nk, nc, nv, nk, nc, nv), dtype=complex)

    one_over_E_diff_cond = one_over_E_diff(Econd)
    one_over_E_diff_val = one_over_E_diff(Eval)

    t1 = np.einsum('kPvKCV,kcP,kcP->kcvKCV', kernel_matrix, elph_cond[imode], one_over_E_diff_cond)
    t2 = np.einsum('kcvKPV,KPC,KCP->kcvKCV', kernel_matrix, elph_cond[imode], one_over_E_diff_cond)
    t3 = np.einsum('kcPKCV,kvP,kvP->kcvKCV', kernel_matrix, elph_val[imode], one_over_E_diff_val)
    t4 = np.einsum('kcvKCP,KPV,KVP->kcvKCV', kernel_matrix, elph_val[imode], one_over_E_diff_val)

    DKernel_dr_imode = t1 + t2 + t3 + t4

    return DKernel_dr_imode


def apply_Qshift_on_valence_states(Qshift, Gv, Kpoints_in_elph_file_frac, verbose=True):
    if np.linalg.norm(Qshift) > 0.0:
        if verbose:
            print(f"\nApplying Q shift to valence states (from finite momentum BSE calculation)")
        Kpoints_shifted = Kpoints_in_elph_file_frac + Qshift
        Kpoints_shifted = np.round(Kpoints_shifted, decimals=6)
        Kpoints_shifted = Kpoints_shifted % 1.0

        mapping = []
        for kshifted in Kpoints_shifted:
            distances = np.linalg.norm(Kpoints_in_elph_file_frac - kshifted, axis = 1)
            min_index = np.argmin(distances)
            if distances[min_index] > 1e-4:
                print(f"WARNING! The Q-shifted k point {kshifted} is not close to any k point in the DFPT calculation. The closest one is {Kpoints_in_elph_file_frac[min_index]} with distance {distances[min_index]}")
            mapping.append(min_index)

        Gv = Gv[:, mapping, :, :, :, :]
        if verbose:
            print(f"Done applying Q shift to valence states")
    else:
        if verbose:
            print(f"\nNOT applying Q shift to valence states (from finite momentum BSE calculation).")
    return Gv


def compute_A_dRPA_dr_imode_B_not_vectorized(Akcv, Bkcv, elph_cond, elph_val):
    Nk, Nc, Nv = Akcv.shape[0], Akcv.shape[1], Akcv.shape[2]
    Nmodes = elph_cond.shape[0]
    forces_imode = np.zeros((Nmodes), dtype=complex)

    for imode in range(Nmodes):
        temp = 0.0 + 0.0j

        for ik in range(Nk):

            for iv in range(Nv):
                for ic1 in range(Nc):
                    for ic2 in range(Nc):
                        temp += Akcv[ik, ic1, iv] * np.conj(Bkcv[ik, ic2, iv]) * elph_cond[imode, ik, ic1, ic2]

            for ic in range(Nc):
                for iv1 in range(Nv):
                    for iv2 in range(Nv):
                        temp -= Akcv[ik, ic, iv1] * np.conj(Bkcv[ik, ic, iv2]) * elph_val[imode, ik, iv1, iv2]

        forces_imode[imode] = temp

    return forces_imode


def compute_A_dRPA_dr_imode_B_vectorized(Akcv, Bkcv, dRPA_dr_imode_mat):
    Nk, Nc, Nv = Akcv.shape[0], Akcv.shape[1], Akcv.shape[2]
    Nmodes = dRPA_dr_imode_mat.shape[0]
    forces_imode = np.zeros((Nmodes), dtype=complex)

    A_mat = np.einsum('abc,ade->abcde', Akcv, np.conj(Bkcv))

    for imode in range(Nmodes):
        temp = np.einsum('abcde,abcde->', A_mat, dRPA_dr_imode_mat[imode])
        forces_imode[imode] = np.sum(temp)

    return forces_imode


def compute_A_dRPAdiag_dr_imode_B_not_vectorized(Akcv, Bkcv, elph_cond, elph_val):
    Nk, Nc, Nv = Akcv.shape[0], Akcv.shape[1], Akcv.shape[2]
    Nmodes = elph_cond.shape[0]
    forces_imode = np.zeros((Nmodes), dtype=complex)

    for imode in range(Nmodes):
        temp = 0.0 + 0.0j
        for ik in range(Nk):
            for ic in range(Nc):
                for iv in range(Nv):
                    temp += np.conj(Akcv[ik, ic, iv]) * Bkcv[ik, ic, iv] * (elph_cond[imode, ik, ic, ic] - elph_val[imode, ik, iv, iv])
        forces_imode[imode] = temp

    return forces_imode


def compute_A_dRPAdiag_dr_imode_B_vectorized(Akcv, Bkcv, dRPA_dr_imode_diag_mat):
    Nk, Nc, Nv = Akcv.shape[0], Akcv.shape[1], Akcv.shape[2]
    Nmodes = dRPA_dr_imode_diag_mat.shape[0]
    forces_imode = np.zeros((Nmodes), dtype=complex)

    A_mat = np.einsum('abc,abc->abc', np.conj(Akcv), Bkcv)

    for imode in range(Nmodes):
        temp = np.einsum('abc,abc->', A_mat, dRPA_dr_imode_diag_mat[imode])
        forces_imode[imode] = np.sum(temp)

    return forces_imode


def compute_A_dKernel_dr_imode_B_not_vectorized(Akcv, Bkcv, DKernel_dr_imode_mat):
    Nk, Nc, Nv = Akcv.shape[0], Akcv.shape[1], Akcv.shape[2]
    Nmodes = DKernel_dr_imode_mat.shape[0]
    forces_imode = np.zeros((Nmodes), dtype=complex)

    for imode in range(Nmodes):

        temp_sum = 0.0 + 0.0j
        for ik1 in range(Nk):
            for ic1 in range(Nc):
                for iv1 in range(Nv):
                    for ik2 in range(Nk):
                        for ic2 in range(Nc):
                            for iv2 in range(Nv):
                                temp_sum += np.conj(Akcv[ik1, ic1, iv1]) * DKernel_dr_imode_mat[imode, ik1, ic1, iv1, ik2, ic2, iv2] * Bkcv[ik2, ic2, iv2]

        forces_imode[imode] = temp_sum

    return forces_imode


def compute_A_dKernel_dr_imode_B_vectorized(Akcv, Bkcv, DKernel_dr_imode_mat):
    Nmodes = DKernel_dr_imode_mat.shape[0]
    forces_imode = np.zeros((Nmodes), dtype=complex)

    A_mat = np.einsum('abc,def->abcdef', np.conj(Akcv), Bkcv)

    for imode in range(Nmodes):
        temp = np.einsum('abcdef,abcdef->', A_mat, DKernel_dr_imode_mat[imode])
        forces_imode[imode] = np.sum(temp)

    return forces_imode


def compute_A_dRPA_dr_imode_B(Akcv, Bkcv, dRPA_dr_imode_mat, elph_cond, elph_val, vectorized=True):
    if vectorized:
        return compute_A_dRPA_dr_imode_B_vectorized(Akcv, Bkcv, dRPA_dr_imode_mat)
    else:
        return compute_A_dRPA_dr_imode_B_not_vectorized(Akcv, Bkcv, elph_cond, elph_val)


def compute_A_dRPAdiag_dr_imode_B(Akcv, Bkcv, dRPA_dr_imode_diag_mat, elph_cond, elph_val, vectorized=True):
    if vectorized:
        return compute_A_dRPAdiag_dr_imode_B_vectorized(Akcv, Bkcv, dRPA_dr_imode_diag_mat)
    else:
        return compute_A_dRPAdiag_dr_imode_B_not_vectorized(Akcv, Bkcv, elph_cond, elph_val)


def compute_A_dKernel_dr_imode_B(Akcv, Bkcv, DKernel_dr_imode_mat, vectorized=True):
    if vectorized:
        return compute_A_dKernel_dr_imode_B_vectorized(Akcv, Bkcv, DKernel_dr_imode_mat)
    else:
        return compute_A_dKernel_dr_imode_B_not_vectorized(Akcv, Bkcv, DKernel_dr_imode_mat)
