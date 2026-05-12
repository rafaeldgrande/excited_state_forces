
import sys
import argparse
import time
import numpy as np
import h5py
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import _gb, read_eqp_dat_file, rec_cm_to_eV

def delta_E(Econd, Eval):
    # Econd: (nc, nk), Eval: (nv, nk) → output: (nk, nc, nv)
    return Econd.T[:, :, np.newaxis] - Eval.T[:, np.newaxis, :]

def calculate_tensor_first_order_not_vectorized(ialpha, ibeta):
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    t0 = time.perf_counter()
    sum_temp = np.zeros((Nmodes, Nfreq), dtype=complex)

    for imode in range(Nmodes):
        for ik in range(nk_elph):
            for ic in range(nc_elph):
                for iv in range(nv_elph):

                    term1_cond = pa[ik, ic, iv] / (Ex - DeltaE[ik, ic, iv] - 1j*gamma)
                    term2_cond = np.zeros(Nfreq, dtype=complex)
                    for icp in range(nc_elph):
                        term2_cond += pb[ik, icp, iv] * g_cond[imode, ik, ic, icp] / (Ex - DeltaE[ik, icp, iv] - freqs_eV[imode] - 1j*gamma)
                    sum_temp[imode] += term1_cond * term2_cond

                    temp1_val = pa[ik, ic, iv] / (Ex - DeltaE[ik, ic, iv] - 1j*gamma)
                    temp2_val = np.zeros(Nfreq, dtype=complex)
                    for ivp in range(nv_elph):
                        temp2_val += pb[ik, ic, ivp] * g_val[imode, ik, iv, ivp] / (Ex - DeltaE[ik, ic, ivp] - freqs_eV[imode] - 1j*gamma)
                    sum_temp[imode] -= temp1_val * temp2_val

        elapsed = time.perf_counter() - t0
        done = imode + 1
        eta = elapsed / done * (Nmodes - done)
        print(f'  mode {done}/{Nmodes} (alpha={ialpha+1}, beta={ibeta+1}) elapsed={elapsed:.1f}s ETA={eta:.1f}s')

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f}s')
    return sum_temp

def calculate_tensor_second_order_not_vectorized(ialpha, ibeta):
    
    # by now just doing calculations with q = Gamma
    
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]
    
    t0 = time.perf_counter()
    iter_done = 0
    total_iters = Nmodes * Nmodes
    sum_temp = np.zeros((Nmodes, Nmodes, Nfreq), dtype=complex)

    for imode in range(Nmodes):
        for jmode in range(Nmodes):
            for ik in range(nk_elph):
                for ic in range(nc_elph):
                    for iv in range(nv_elph):
                        
                        term1_cond = pa[ik, ic, iv] / (Ex - DeltaE[ik, ic, iv] - 1j*gamma)
                        sum_term2_cond = np.zeros(Nfreq, dtype=complex)
                        for icp in range(nc_elph):
                            term2_cond = g_cond[imode, ik, ic, icp] / (Ex - DeltaE[ik, icp, iv] - freqs_eV[imode] - 1j*gamma)
                            sum_term3_cond = np.zeros(Nfreq, dtype=complex)
                            for icpp in range(nc_elph):
                                sum_term3_cond += pb[ik, icpp, iv] * g_cond[jmode, ik, icp, icpp] / (Ex - DeltaE[ik, icpp, iv] - freqs_eV[jmode] - freqs_eV[imode] - 1j*gamma)
                            sum_term2_cond += term2_cond * sum_term3_cond
                        sum_temp[imode, jmode] += term1_cond * sum_term2_cond
                        
                        term1_val = pa[ik, ic, iv] / (Ex - DeltaE[ik, ic, iv] - 1j*gamma)
                        sum_term2_val = np.zeros(Nfreq, dtype=complex)
                        for ivp in range(nv_elph):
                            term2_val = g_val[imode, ik, iv, ivp] / (Ex - DeltaE[ik, ic, ivp] - freqs_eV[imode] - 1j*gamma)
                            sum_term3_val = np.zeros(Nfreq, dtype=complex)
                            for ivpp in range(nv_elph):
                                sum_term3_val += pb[ik, ic, ivpp] * g_val[jmode, ik, ivp, ivpp] / (Ex - DeltaE[ik, ic, ivpp] - freqs_eV[jmode] - freqs_eV[imode] - 1j*gamma)
                            sum_term2_val += term2_val * sum_term3_val
                        sum_temp[imode, jmode] += term1_val * sum_term2_val
                        
                        term1_mixed = pa[ik, ic, iv] / (Ex - DeltaE[ik, ic, iv] - 1j*gamma)
                        sum_term2_mixed = np.zeros(Nfreq, dtype=complex)
                        for icp in range(nc_elph):
                            term2_mixed = g_cond[imode, ik, ic, icp] / (Ex - DeltaE[ik, icp, iv] - freqs_eV[imode] - 1j*gamma)
                            sum_term3_mixed = np.zeros(Nfreq, dtype=complex)
                            for ivp in range(nv_elph):
                                sum_term3_mixed += pb[ik, icp, ivp] * g_val[jmode, ik, ivp, iv] / (Ex - DeltaE[ik, icp, ivp] - freqs_eV[jmode] - freqs_eV[imode] - 1j*gamma)
                            sum_term2_mixed += term2_mixed * sum_term3_mixed
                        sum_temp[imode, jmode] -= term1_mixed * sum_term2_mixed

            iter_done += 1
            elapsed = time.perf_counter() - t0
            eta = elapsed / iter_done * (total_iters - iter_done) if iter_done < total_iters else 0.0
            print(f'  ({imode+1},{jmode+1})/({Nmodes},{Nmodes}) (alpha={ialpha+1}, beta={ibeta+1}) elapsed={elapsed:.1f}s ETA={eta:.1f}s')

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f}s')
    return sum_temp


def calculate_tensor_second_order_vectorized_over_kcv(ialpha, ibeta):
    """Second-order tensor vectorized over k, c, v; both mode loops kept in Python."""
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    t0 = time.perf_counter()
    iter_done = 0
    total_iters = Nmodes * Nmodes
    sum_temp = np.zeros((Nmodes, Nmodes, Nfreq), dtype=complex)

    G1 = pa[:, :, :, None] / (Ex - DeltaE[:, :, :, None] - 1j * gamma)  # (nk, nc, nv, Nf)

    for imode in range(Nmodes):
        inv_D2_i = 1 / (Ex - DeltaE[:, :, :, None] - freqs_eV[imode] - 1j * gamma)

        for jmode in range(Nmodes):

            inv_D3 = 1 / (Ex - DeltaE[:, :, :, None] - freqs_eV[imode] - freqs_eV[jmode] - 1j * gamma)
            pb_inv_D3 = pb[:, :, :, None] * inv_D3  # (nk, nc, nv, Nf)

            # pb_inv_D3 transposed for val and mixed contractions over nv
            pb_inv_D3_T = pb_inv_D3.transpose(0, 2, 1, 3).reshape(nk_elph, nv_elph, -1)  # (nk, nv, nc*Nf)

            # Conduction: Σ_{c'} g_cond[i] Σ_{c''} g_cond[j] * pb_inv_D3[c''] * inv_D2_i[c']
            T3c = (np.matmul(g_cond[jmode], pb_inv_D3.reshape(nk_elph, nc_elph, -1))
                     .reshape(nk_elph, nc_elph, nv_elph, Nfreq))
            A_c = T3c * inv_D2_i
            T2c = (np.matmul(g_cond[imode], A_c.reshape(nk_elph, nc_elph, -1))
                     .reshape(nk_elph, nc_elph, nv_elph, Nfreq))
            sum_temp[imode, jmode] += (G1 * T2c).sum(axis=(0, 1, 2))

            # Valence: Σ_{v'} g_val[i] Σ_{v''} g_val[j] * pb_inv_D3[v''] * inv_D2_i[v']
            T3v = (np.matmul(g_val[jmode], pb_inv_D3_T)
                     .reshape(nk_elph, nv_elph, nc_elph, Nfreq)
                     .transpose(0, 2, 1, 3))                   # (nk, nc, nv', Nf)
            A_v = T3v * inv_D2_i
            A_v_T = A_v.transpose(0, 2, 1, 3).reshape(nk_elph, nv_elph, -1)
            T2v = (np.matmul(g_val[imode], A_v_T)
                     .reshape(nk_elph, nv_elph, nc_elph, Nfreq)
                     .transpose(0, 2, 1, 3))                   # (nk, nc, nv, Nf)
            sum_temp[imode, jmode] += (G1 * T2v).sum(axis=(0, 1, 2))

            # Mixed: Σ_{c'} g_cond[i] Σ_{v'} g_val[j,v',v] * pb_inv_D3[c',v'] * inv_D2_i[c',v]
            gvalT_j = g_val[jmode].transpose(0, 2, 1)          # (nk, nv_outer, nv')
            T3m = (np.matmul(gvalT_j, pb_inv_D3_T)
                     .reshape(nk_elph, nv_elph, nc_elph, Nfreq)
                     .transpose(0, 2, 1, 3))                   # (nk, nc', nv_outer, Nf)
            A_m = T3m * inv_D2_i
            T2m = (np.matmul(g_cond[imode], A_m.reshape(nk_elph, nc_elph, -1))
                     .reshape(nk_elph, nc_elph, nv_elph, Nfreq))
            sum_temp[imode, jmode] -= (G1 * T2m).sum(axis=(0, 1, 2))

            iter_done += 1
            elapsed = time.perf_counter() - t0
            eta = elapsed / iter_done * (total_iters - iter_done) if iter_done < total_iters else 0.0
            print(f'  ({imode+1},{jmode+1})/({Nmodes},{Nmodes}) (alpha={ialpha+1}, beta={ibeta+1}) elapsed={elapsed:.1f}s ETA={eta:.1f}s')

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f}s')
    return sum_temp


def calculate_tensor_second_order_vectorized_over_jmode_and_kcv(ialpha, ibeta):
    """Second-order tensor vectorized over j-modes, k, c, v; imode loop kept in Python."""
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    t0 = time.perf_counter()
    sum_temp = np.zeros((Nmodes, Nmodes, Nfreq), dtype=complex)

    G1 = pa[:, :, :, None] / (Ex - DeltaE[:, :, :, None] - 1j * gamma)  # (nk, nc, nv, Nf)
    gvalT_all = g_val.transpose(0, 1, 3, 2)  # (Nmodes, nk, nv, nv) — transposed for mixed term

    for imode in range(Nmodes):

        inv_D2_i = 1 / (Ex - DeltaE[:, :, :, None] - freqs_eV[imode] - 1j * gamma)

        # inv_D3_all[j, k, c, v, f]   shape (Nmodes, nk, nc, nv, Nfreq)
        inv_D3_all = 1 / (Ex
                          - DeltaE[None, :, :, :, None]
                          - freqs_eV[imode]
                          - freqs_eV[:, None, None, None, None]
                          - 1j * gamma)

        pb_inv_D3_all = pb[None, :, :, :, None] * inv_D3_all  # (Nmodes, nk, nc, nv, Nf)
        pb_inv_D3_T_all = pb_inv_D3_all.transpose(0, 1, 3, 2, 4).reshape(Nmodes, nk_elph, nv_elph, -1)

        # Conduction
        T3c_all = (np.matmul(g_cond, pb_inv_D3_all.reshape(Nmodes, nk_elph, nc_elph, -1))
                     .reshape(Nmodes, nk_elph, nc_elph, nv_elph, Nfreq))
        A_c_all = T3c_all * inv_D2_i[None]
        T2c_all = (np.matmul(g_cond[imode][None], A_c_all.reshape(Nmodes, nk_elph, nc_elph, -1))
                     .reshape(Nmodes, nk_elph, nc_elph, nv_elph, Nfreq))
        sum_temp[imode] += (G1[None] * T2c_all).sum(axis=(1, 2, 3))  # (Nmodes, Nf)

        # Valence
        T3v_all = (np.matmul(g_val, pb_inv_D3_T_all)
                     .reshape(Nmodes, nk_elph, nv_elph, nc_elph, Nfreq)
                     .transpose(0, 1, 3, 2, 4))                        # (Nm, nk, nc, nv', Nf)
        A_v_all = T3v_all * inv_D2_i[None]
        A_v_T_all = A_v_all.transpose(0, 1, 3, 2, 4).reshape(Nmodes, nk_elph, nv_elph, -1)
        T2v_all = (np.matmul(g_val[imode][None], A_v_T_all)
                     .reshape(Nmodes, nk_elph, nv_elph, nc_elph, Nfreq)
                     .transpose(0, 1, 3, 2, 4))                        # (Nm, nk, nc, nv, Nf)
        sum_temp[imode] += (G1[None] * T2v_all).sum(axis=(1, 2, 3))

        # Mixed
        T3m_all = (np.matmul(gvalT_all, pb_inv_D3_T_all)
                     .reshape(Nmodes, nk_elph, nv_elph, nc_elph, Nfreq)
                     .transpose(0, 1, 3, 2, 4))                        # (Nm, nk, nc', nv_outer, Nf)
        A_m_all = T3m_all * inv_D2_i[None]
        T2m_all = (np.matmul(g_cond[imode][None], A_m_all.reshape(Nmodes, nk_elph, nc_elph, -1))
                     .reshape(Nmodes, nk_elph, nc_elph, nv_elph, Nfreq))
        sum_temp[imode] -= (G1[None] * T2m_all).sum(axis=(1, 2, 3))

        elapsed = time.perf_counter() - t0
        done = imode + 1
        eta = elapsed / done * (Nmodes - done)
        print(f'  imode {done}/{Nmodes} (alpha={ialpha+1}, beta={ibeta+1}) elapsed={elapsed:.1f}s ETA={eta:.1f}s')

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f}s')
    return sum_temp


def calculate_tensor_first_order_vectorized_over_kcv(ialpha, ibeta):
    """Vectorized over k, c, v; mode loop kept in Python."""
    pa = pos_operator_list[ialpha]  # (nk, nc, nv)
    pb = pos_operator_list[ibeta]

    ram = _gb(
        (nk_elph, nc_elph, nv_elph, Nfreq), complex,  # G1
        (nk_elph, nc_elph, nv_elph, Nfreq), complex,  # inv_D2
        (nk_elph, nc_elph, nv_elph, Nfreq), complex,  # pb_inv_D2
        (nk_elph, nc_elph, nv_elph, Nfreq), complex,  # cond_part
        (nk_elph, nc_elph, nv_elph, Nfreq), complex,  # val_part
        (Nmodes, Nfreq),                               complex,  # sum_temp
    )
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — G1, inv_D2, pb_inv_D2, cond_part, val_part, sum_temp — {ram:.3f} GB')

    t0 = time.perf_counter()
    sum_temp = np.zeros((Nmodes, Nfreq), dtype=complex)

    # G1[k, c, v, f] = pa[k,c,v] / (Ex[f] - DeltaE[k,c,v] - iγ)   shape (nk, nc, nv, Nfreq)
    G1 = pa[:, :, :, np.newaxis] / (Ex - DeltaE[:, :, :, np.newaxis] - 1j * gamma)

    for imode in range(Nmodes):

        # inv_D2[k, c, v, f] = 1 / (Ex[f] - DeltaE[k,c,v] - ω_m - iγ)   (nk, nc, nv, Nfreq)
        inv_D2 = 1 / (Ex - DeltaE[:, :, :, np.newaxis] - freqs_eV[imode] - 1j * gamma)

        # pb_inv_D2[k, c', v, f] = pb[k, c', v] * inv_D2[k, c', v, f]   (nk, nc, nv, Nfreq)
        pb_inv_D2 = pb[:, :, :, np.newaxis] * inv_D2

        # --- Conduction: Σ_{c'} g_cond[m,k,c,c'] * pb_inv_D2[k,c',v,f] ---
        # g_cond[imode] (nk, nc, nc) @ pb_inv_D2 flat (nk, nc, nv*Nf) → (nk, nc, nv*Nf)
        cond_part = (np.matmul(g_cond[imode], pb_inv_D2.reshape(nk_elph, nc_elph, -1))
                       .reshape(nk_elph, nc_elph, nv_elph, Nfreq))
        sum_temp[imode] += (G1 * cond_part).sum(axis=(0, 1, 2))

        # --- Valence: Σ_{v'} g_val[m,k,v,v'] * pb_inv_D2[k,c,v',f] ---
        # Permute pb_inv_D2 → (nk, nv', nc, Nf) → (nk, nv', nc*Nf) for batched matmul
        # g_val[imode] (nk, nv, nv') @ (nk, nv', nc*Nf) → (nk, nv, nc*Nf)
        # → reshape (nk, nv, nc, Nf) → transpose (nk, nc, nv, Nf)
        pb_inv_D2_t = pb_inv_D2.transpose(0, 2, 1, 3).reshape(nk_elph, nv_elph, -1)
        val_part = (np.matmul(g_val[imode], pb_inv_D2_t)
                      .reshape(nk_elph, nv_elph, nc_elph, Nfreq)
                      .transpose(0, 2, 1, 3))
        sum_temp[imode] -= (G1 * val_part).sum(axis=(0, 1, 2))

        elapsed = time.perf_counter() - t0
        done = imode + 1
        eta = elapsed / done * (Nmodes - done)
        print(f'  mode {done}/{Nmodes} (alpha={ialpha+1}, beta={ibeta+1}) elapsed={elapsed:.1f}s ETA={eta:.1f}s')

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f}s')
    return sum_temp


def calculate_tensor_first_order_vectorized_over_modes_and_kcv(ialpha, ibeta):
    """Fully vectorized over modes, k, c, v. Higher memory but no Python mode loop."""
    pa = pos_operator_list[ialpha]  # (nk, nc, nv)
    pb = pos_operator_list[ibeta]

    ram = _gb(
        (nk_elph, nc_elph, nv_elph, Nfreq),          complex,  # G1
        (Nmodes, nk_elph, nc_elph, nv_elph, Nfreq),  complex,  # inv_D2_all
        (Nmodes, nk_elph, nc_elph, nv_elph, Nfreq),  complex,  # pb_inv_D2_all
        (Nmodes, nk_elph, nc_elph, nv_elph, Nfreq),  complex,  # cond_parts
        (Nmodes, nk_elph, nc_elph, nv_elph, Nfreq),  complex,  # val_parts
        (Nmodes, Nfreq),                               complex,  # sum_temp
    )
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — G1, inv_D2_all, pb_inv_D2_all, cond_parts, val_parts, sum_temp — {ram:.3f} GB')

    t0 = time.perf_counter()
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) vectorized over modes and k,c,v')

    # G1[k, c, v, f] = pa[k,c,v] / (Ex[f] - DeltaE[k,c,v] - iγ)   (nk, nc, nv, Nfreq)
    G1 = pa[:, :, :, np.newaxis] / (Ex - DeltaE[:, :, :, np.newaxis] - 1j * gamma)

    # inv_D2_all[m, k, c, v, f] = 1 / (Ex[f] - DeltaE[k,c,v] - ω_m - iγ)   (Nmodes, nk, nc, nv, Nfreq)
    inv_D2_all = 1 / (Ex
                      - DeltaE[np.newaxis, :, :, :, np.newaxis]
                      - freqs_eV[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
                      - 1j * gamma)

    # pb_inv_D2_all[m, k, c', v, f] = pb[k, c', v] * inv_D2_all[m, k, c', v, f]   (Nmodes, nk, nc, nv, Nfreq)
    pb_inv_D2_all = pb[np.newaxis, :, :, :, np.newaxis] * inv_D2_all

    # --- Conduction: Σ_{c'} g_cond[m,k,c,c'] * pb_inv_D2_all[m,k,c',v,f] ---
    # g_cond (Nmodes, nk, nc, nc) @ pb_inv_D2_all flat (Nmodes, nk, nc, nv*Nf)
    # → (Nmodes, nk, nc, nv*Nf) → reshape (Nmodes, nk, nc, nv, Nfreq)
    cond_parts = (np.matmul(g_cond, pb_inv_D2_all.reshape(Nmodes, nk_elph, nc_elph, -1))
                    .reshape(Nmodes, nk_elph, nc_elph, nv_elph, Nfreq))
    sum_temp = (G1[np.newaxis] * cond_parts).sum(axis=(1, 2, 3))  # (Nmodes, Nfreq)

    # --- Valence: Σ_{v'} g_val[m,k,v,v'] * pb_inv_D2_all[m,k,c,v',f] ---
    # Permute pb_inv_D2_all → (Nmodes, nk, nv', nc, Nf) → (Nmodes, nk, nv', nc*Nf)
    # g_val (Nmodes, nk, nv, nv') @ (Nmodes, nk, nv', nc*Nf) → (Nmodes, nk, nv, nc*Nf)
    # → reshape (Nmodes, nk, nv, nc, Nf) → transpose (Nmodes, nk, nc, nv, Nf)
    pb_inv_D2_t = pb_inv_D2_all.transpose(0, 1, 3, 2, 4).reshape(Nmodes, nk_elph, nv_elph, -1)
    val_parts = (np.matmul(g_val, pb_inv_D2_t)
                   .reshape(Nmodes, nk_elph, nv_elph, nc_elph, Nfreq)
                   .transpose(0, 1, 3, 2, 4))
    sum_temp -= (G1[np.newaxis] * val_parts).sum(axis=(1, 2, 3))

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return sum_temp


parser = argparse.ArgumentParser()
parser.add_argument('--elph_fine_file', default='elph_fine.h5',
                    help='HDF5 file from interpolate_elph_bgw.py (default: elph_fine.h5)')
parser.add_argument('--eqp_file', default='eqp.dat',
                    help='Fallback eqp.dat for QP energies when not stored in --elph_fine_file '
                         '(default: eqp.dat)')
parser.add_argument('--nval_in_eqp', type=int, default=-1,
                    help='Index of highest valence band in eqp.dat; only used with eqp.dat fallback '
                         '(default: -1)')
parser.add_argument('--no_renorm_elph', action='store_true',
                    help='Skip QP renormalization of el-ph coefficients')
parser.add_argument('--dip_mom_noeh_file_b1', default='eigenvalues_b1_noeh.dat', help='Dipole moment file for polarization b1 (default: eigenvalues_b1_noeh.dat)')
parser.add_argument('--dip_mom_noeh_file_b2', default='eigenvalues_b2_noeh.dat', help='Dipole moment file for polarization b2 (default: eigenvalues_b2_noeh.dat)')
parser.add_argument('--dip_mom_noeh_file_b3', default='eigenvalues_b3_noeh.dat', help='Dipole moment file for polarization b3 (default: eigenvalues_b3_noeh.dat)')
parser.add_argument('--dE', type=float, default=0.001, help='Energy step in eV for the Ex grid')
parser.add_argument('--Emin', type=float, default=0.0, help='Minimum excitation energy in eV')
parser.add_argument('--Emax', type=float, default=10.0, help='Maximum excitation energy in eV')
parser.add_argument('--gamma', type=float, default=0.01, help='Broadening parameter in eV')
parser.add_argument('--vectorized_flavor', type=int, default=2, choices=[0, 1, 2], help='0: no vectorization (triple loop), 1: vectorize over excitons only, 2: vectorize over both excitons and modes (more memory usage but faster; default: 2)')
parser.add_argument('--compute_second_order', action='store_true', help='Compute and save the second-order susceptibility tensor')
parser.add_argument('--vectorized_flavor_second_order', type=int, default=1, choices=[0, 1, 2], help='0: no vectorization, 1: vectorize over k,c,v (jmode loop kept), 2: vectorize over jmode+k,c,v (more memory; default: 1)')
parser.add_argument('--test_functions', action='store_true', help='Run all three implementations on truncated data and check they agree')
parser.add_argument('--freqs_file', default=None,
                    help='Fallback phonon frequencies file in cm^-1; only used when not in '
                         '--elph_fine_file (default: None)')
parser.add_argument('--write_dummy', action='store_true',
                    help='Also compute and save dummy tensors (all numerators = 1, joint DOS of transitions) '
                         'to susceptibility_tensors_first_order_dummy.h5')
parser.add_argument('--limit_transitions', type=int, default=None,
                    help='Truncate to first N transitions (for quick tests; default: None)')
parser.add_argument('--flavor_energy_levels', type=int, default=1, choices=[1, 2], help='Flavor energy levels. 1 = GW, 2 = DFT (default: 1)')
args = parser.parse_args()

flavor_energy_levels = args.flavor_energy_levels
elph_fine_file = args.elph_fine_file
dip_mom_noeh_file_b1 = args.dip_mom_noeh_file_b1
dip_mom_noeh_file_b2 = args.dip_mom_noeh_file_b2
dip_mom_noeh_file_b3 = args.dip_mom_noeh_file_b3
vectorized_flavor = args.vectorized_flavor
compute_second_order = args.compute_second_order
vectorized_flavor_second_order = args.vectorized_flavor_second_order
test_functions    = args.test_functions
freqs_file = args.freqs_file
dE = args.dE
Emin = args.Emin
Emax = args.Emax
gamma = args.gamma  # eV
limit_transitions = args.limit_transitions
eqp_file = args.eqp_file
nval_in_eqp = args.nval_in_eqp
no_renorm_elph = args.no_renorm_elph

flavor_desc = {0: 'no vectorization (quintuple loop)',
               1: 'vectorized over k, c, v',
               2: 'vectorized over modes and k, c, v'}
flavor_desc_2nd = {0: 'no vectorization (septuple loop)',
                   1: 'vectorized over k, c, v (jmode loop kept)',
                   2: 'vectorized over jmode + k, c, v'}
print('--- Options ---')
print(f'  elph_fine_file    : {elph_fine_file}')
print(f'  eqp_file          : {eqp_file}  (fallback if QP data not in h5)')
print(f'  nval_in_eqp       : {nval_in_eqp}  (used only with eqp.dat fallback)')
print(f'  no_renorm_elph    : {no_renorm_elph}')
print(f'  freqs_file        : {freqs_file}  (fallback if freqs not in h5)')
print(f'  dip_mom_file_b1   : {dip_mom_noeh_file_b1}')
print(f'  dip_mom_file_b2   : {dip_mom_noeh_file_b2}')
print(f'  dip_mom_file_b3   : {dip_mom_noeh_file_b3}')
print(f'  dE                : {dE} eV')
print(f'  Emin              : {Emin} eV')
print(f'  Emax              : {Emax} eV')
print(f'  gamma             : {gamma} eV')
print(f'  vectorized_flavor : {vectorized_flavor} ({flavor_desc[vectorized_flavor]})')
print(f'  compute_second_order          : {compute_second_order}')
print(f'  vectorized_flavor_second_order: {vectorized_flavor_second_order} ({flavor_desc_2nd[vectorized_flavor_second_order]})')
print(f'  test_functions    : {test_functions}')
print(f'  write_dummy       : {args.write_dummy}')
print(f'  flavor_energy_levels : {flavor_energy_levels}. 1 = GW, 2 = DFT')
print('---------------\n')

# Energy grid
Ex = np.arange(Emin, Emax, dE)  # eV
Nfreq = Ex.shape[0]

# ---------------------------------------------------------------------------
# 1-3. Load from elph_fine.h5 (interpolate_elph_bgw.py output)
# ---------------------------------------------------------------------------
_TOL_Q = 1e-5
Eqp_cond = Eqp_val = Edft_cond = Edft_val = None
QP_rescaling_cond = QP_rescaling_val = None
phonon_frequencies = None

print(f'\nLoading fine-grid el-ph from {elph_fine_file}')
with h5py.File(elph_fine_file, 'r') as hf:
    # Find q=0 (Gamma) in the q-point list
    qpoints_crystal = hf['qpoints_crystal'][:]       # (Nq, 3) fractional coords
    iq0 = next(
        (i for i, qc in enumerate(qpoints_crystal)
         if np.linalg.norm(qc - np.round(qc)) < _TOL_Q), -1)
    if iq0 == -1:
        sys.exit(f'ERROR: q=0 (Gamma) not found in {elph_fine_file} qpoints_crystal.')
    print(f'  q=0 (Gamma): index iq={iq0}  (q_crystal = {qpoints_crystal[iq0]})')

    g_cond = hf['elph_fine_cond_mode'][iq0].astype(complex)  # (Nmodes, Nk, Nc, Nc)
    g_val  = hf['elph_fine_val_mode'][iq0].astype(complex)   # (Nmodes, Nk, Nv, Nv)

    # Phonon frequencies from phonon_modes group (same q ordering as elph arrays)
    if 'phonon_modes/frequencies' in hf:
        phonon_frequencies = hf['phonon_modes/frequencies'][iq0]  # (Nmodes,) cm^-1

    # QP rescaling matrices and band energies
    if 'QP_rescaling_matrix_cond' in hf:
        QP_rescaling_cond = hf['QP_rescaling_matrix_cond'][:]   # (Nk, Nc, Nc)
        QP_rescaling_val  = hf['QP_rescaling_matrix_val'][:]    # (Nk, Nv, Nv)
        # stored as (Nk, Nb); transpose to (Nb, Nk) for delta_E compatibility
        Eqp_cond  = hf['Eqp_cond'][:].T                        # (Nc, Nk)
        Eqp_val   = hf['Eqp_val'][:].T                         # (Nv, Nk)
        Edft_cond = hf['Edft_cond'][:].T if 'Edft_cond' in hf else None
        Edft_val  = hf['Edft_val'][:].T  if 'Edft_val'  in hf else None
        print(f'  Loaded QP rescaling matrices and energies from {elph_fine_file}')

# Fall back to external files if not found in h5
if phonon_frequencies is None:
    if freqs_file is None:
        sys.exit('ERROR: phonon_modes/frequencies not in h5 and --freqs_file not given.')
    phonon_frequencies = np.loadtxt(freqs_file)
    print(f'  Phonon frequencies: read from {freqs_file}')

freqs_rec_cm = phonon_frequencies
freqs_eV     = freqs_rec_cm * rec_cm_to_eV
Nmodes       = len(freqs_eV)
print(f'  Phonon modes: {Nmodes}  (max freq = {freqs_rec_cm.max():.1f} cm^-1)')

nk_elph = g_cond.shape[1]
nc_elph = g_cond.shape[2]
nv_elph = g_val.shape[2]
print(f'  nk={nk_elph}, nv={nv_elph}, nc={nc_elph}')

# QP energies fallback: read eqp.dat when not stored in h5
if Eqp_cond is None:
    print(f'\nQP energies not in h5; reading from {eqp_file}')
    bands_dft, bands_qp, kpoints_eqp, nk_eqp, band_indexes_eqp = read_eqp_dat_file(eqp_file)
    nval_index = np.where(band_indexes_eqp == nval_in_eqp)[0][0]
    Edft_val  = bands_dft[:nval_index+1, :]   # (nv, nk)
    Eqp_val   = bands_qp[:nval_index+1, :]
    Edft_cond = bands_dft[nval_index+1:, :]   # (nc, nk)
    Eqp_cond  = bands_qp[nval_index+1:, :]
    print(f'  nk={Eqp_cond.shape[1]}, nv={Eqp_val.shape[0]}, nc={Eqp_cond.shape[0]}')
else:
    print(f'  Using QP energies from {elph_fine_file}')

if flavor_energy_levels == 1:
    DeltaE = delta_E(Eqp_cond, Eqp_val)
else:
    if Edft_cond is not None:
        DeltaE = delta_E(Edft_cond, Edft_val)
    else:
        print('  WARNING: Edft not available, falling back to Eqp for DeltaE')
        DeltaE = delta_E(Eqp_cond, Eqp_val)
# shape of DeltaE: (nk, nc, nv)

# QP renormalization of el-ph coefficients
def renormalize_elph_coeffs(elph, Eqp_nk_nb, Edft_nk_nb, ratio=None):
    """Scale elph by ΔE_QP/ΔE_DFT for each (k, n, m) pair."""
    if ratio is None:
        dEqp  = Eqp_nk_nb[:, :, None] - Eqp_nk_nb[:, None, :]   # (nk, nb, nb)
        dEdft = Edft_nk_nb[:, :, None] - Edft_nk_nb[:, None, :]
        mask  = np.abs(dEdft) > 1e-6
        ratio = np.ones_like(dEdft)
        ratio[mask] = dEqp[mask] / dEdft[mask]
    return elph * ratio[np.newaxis]   # broadcast over modes axis

if not no_renorm_elph:
    if QP_rescaling_cond is not None:
        g_cond = renormalize_elph_coeffs(g_cond, None, None, QP_rescaling_cond)
        g_val  = renormalize_elph_coeffs(g_val,  None, None, QP_rescaling_val)
        print('  Applied QP renormalization (pre-computed matrices)')
    elif Edft_cond is not None:
        # Eqp/Edft stored as (nb, nk); renorm function needs (nk, nb)
        g_cond = renormalize_elph_coeffs(g_cond, Eqp_cond.T, Edft_cond.T)
        g_val  = renormalize_elph_coeffs(g_val,  Eqp_val.T,  Edft_val.T)
        print('  Applied QP renormalization (from Eqp/Edft energies)')
    else:
        print('  WARNING: Edft not available; skipping QP renormalization')
else:
    print('  Skipping QP renormalization of elph (--no_renorm_elph)')
    
# ---------------------------------------------------------------------------
# 4. Load dipole matrix elements (momentum → position)
# ---------------------------------------------------------------------------


dip_moments_b1_data = np.loadtxt(dip_mom_noeh_file_b1)
dip_moments_b2_data = np.loadtxt(dip_mom_noeh_file_b2)
dip_moments_b3_data = np.loadtxt(dip_mom_noeh_file_b3)

dip_moments_b1 = (dip_moments_b1_data[:, 8] + 1j * dip_moments_b1_data[:, 9]).reshape(nk_elph, nc_elph, nv_elph)
dip_moments_b2 = (dip_moments_b2_data[:, 8] + 1j * dip_moments_b2_data[:, 9]).reshape(nk_elph, nc_elph, nv_elph)
dip_moments_b3 = (dip_moments_b3_data[:, 8] + 1j * dip_moments_b3_data[:, 9]).reshape(nk_elph, nc_elph, nv_elph)

# shape of dip_moments: (nk, nc, nv) is the same as DeltaE
pos_op_b1 = 1j * dip_moments_b1 / DeltaE
pos_op_b2 = 1j * dip_moments_b2 / DeltaE
pos_op_b3 = 1j * dip_moments_b3 / DeltaE

pos_operator_list = [pos_op_b1, pos_op_b2, pos_op_b3]

# ---------------------------------------------------------------------------
# 5. (optional) Test: check all three flavors agree on truncated data
# ---------------------------------------------------------------------------
if test_functions:
    nk_t = min(3, nk_elph)
    nc_t = min(3, nc_elph)
    nv_t = min(2, nv_elph)
    Nm_t = min(4, Nmodes)
    print(f'Test mode: truncating to nk={nk_t}, nc={nc_t}, nv={nv_t}, Nmodes={Nm_t}')

    # Truncate all module-level globals the functions read
    g_cond          = g_cond[:Nm_t, :nk_t, :nc_t, :nc_t]
    g_val           = g_val[:Nm_t,  :nk_t, :nv_t, :nv_t]
    DeltaE          = DeltaE[:nk_t, :nc_t, :nv_t]
    freqs_eV        = freqs_eV[:Nm_t]
    pos_operator_list = [p[:nk_t, :nc_t, :nv_t] for p in pos_operator_list]
    nk_elph, nc_elph, nv_elph, Nmodes = nk_t, nc_t, nv_t, Nm_t

    fns = [
        (0, calculate_tensor_first_order_not_vectorized),
        (1, calculate_tensor_first_order_vectorized_over_kcv),
        (2, calculate_tensor_first_order_vectorized_over_modes_and_kcv),
    ]
    results = {}
    for flavor_id, fn in fns:
        out = np.zeros((3, 3, Nmodes, Nfreq), dtype=complex)
        for ialpha in range(3):
            for ibeta in range(3):
                out[ialpha, ibeta] = fn(ialpha, ibeta)
        results[flavor_id] = out

    ref = results[0]
    all_passed = True
    print('--- First-order ---')
    for flavor_id in [1, 2]:
        err = np.max(np.abs(results[flavor_id] - ref))
        status = 'PASS' if err < 1e-10 else 'FAIL'
        print(f'  1st-order flavor {flavor_id} vs flavor 0 — {status}  (max|diff| = {err:.2e})')
        if status == 'FAIL':
            all_passed = False

    print('--- Second-order ---')
    fns_2nd = [
        (0, calculate_tensor_second_order_not_vectorized),
        (1, calculate_tensor_second_order_vectorized_over_kcv),
        (2, calculate_tensor_second_order_vectorized_over_jmode_and_kcv),
    ]
    results_2nd = {}
    for flavor_id, fn in fns_2nd:
        out = np.zeros((3, 3, Nmodes, Nmodes, Nfreq), dtype=complex)
        for ialpha in range(3):
            for ibeta in range(3):
                out[ialpha, ibeta] = fn(ialpha, ibeta)
        results_2nd[flavor_id] = out

    ref_2nd = results_2nd[0]
    for flavor_id in [1, 2]:
        err = np.max(np.abs(results_2nd[flavor_id] - ref_2nd))
        status = 'PASS' if err < 1e-10 else 'FAIL'
        print(f'  2nd-order flavor {flavor_id} vs flavor 0 — {status}  (max|diff| = {err:.2e})')
        if status == 'FAIL':
            all_passed = False

    print('All tests passed.' if all_passed else 'Some tests FAILED.')
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 6. Calculate susceptibility tensors
# ---------------------------------------------------------------------------

susceptibility_tensor_first_order = np.zeros((3, 3, Nmodes, Nfreq), dtype=complex)

for ialpha in range(3):
    for ibeta in range(3):
        if vectorized_flavor == 2:
            susceptibility_tensor_first_order[ialpha, ibeta] = calculate_tensor_first_order_vectorized_over_modes_and_kcv(ialpha, ibeta)
        elif vectorized_flavor == 1:
            susceptibility_tensor_first_order[ialpha, ibeta] = calculate_tensor_first_order_vectorized_over_kcv(ialpha, ibeta)
        else:
            susceptibility_tensor_first_order[ialpha, ibeta] = calculate_tensor_first_order_not_vectorized(ialpha, ibeta)

output_h5_file = 'susceptibility_tensors_first_order_IPA.h5'
with h5py.File(output_h5_file, 'w') as hf:
    hf.create_dataset('excitation_energies', data=Ex)
    hf.create_dataset('susceptibility_tensor_first_order', data=susceptibility_tensor_first_order)  # (3, 3, Nmodes, Nfreq)
print(f'Saved first-order susceptibility tensors to {output_h5_file}')

if compute_second_order:
    susceptibility_tensor_second_order = np.zeros((3, 3, Nmodes, Nmodes, Nfreq), dtype=complex)

    for ialpha in range(3):
        for ibeta in range(3):
            if vectorized_flavor_second_order == 2:
                susceptibility_tensor_second_order[ialpha, ibeta] = calculate_tensor_second_order_vectorized_over_jmode_and_kcv(ialpha, ibeta)
            elif vectorized_flavor_second_order == 1:
                susceptibility_tensor_second_order[ialpha, ibeta] = calculate_tensor_second_order_vectorized_over_kcv(ialpha, ibeta)
            else:
                susceptibility_tensor_second_order[ialpha, ibeta] = calculate_tensor_second_order_not_vectorized(ialpha, ibeta)

    output_h5_file_2nd = 'susceptibility_tensors_second_order_IPA.h5'
    with h5py.File(output_h5_file_2nd, 'w') as hf:
        hf.create_dataset('excitation_energies', data=Ex)
        hf.create_dataset('susceptibility_tensor_second_order', data=susceptibility_tensor_second_order)  # (3, 3, Nmodes, Nmodes, Nfreq)
    print(f'Saved second-order susceptibility tensors to {output_h5_file_2nd}')
