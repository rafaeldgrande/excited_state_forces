
import sys
import argparse
import os
import time
import numpy as np
import h5py
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import _gb, rec_cm_to_eV

# ---------------------------------------------------------------------------
# Module-level state shared with fork-based worker processes (COW, no copy)
# ---------------------------------------------------------------------------
_mp_D1        = None
_mp_pa_inv_D1 = None
_mp_pb_conj   = None
_mp_so_exc_ph = None
_mp_freqs_eV  = None

def _dbl_worker(imode):
    """Per-mode worker for the double-resonance parallel function."""
    inv_D2    = 1.0 / (_mp_D1 - 2.0 * _mp_freqs_eV[imode])   # (Nexc, Nfreq)
    pb_inv_D2 = _mp_pb_conj[:, np.newaxis] * inv_D2           # (Nexc, Nfreq)
    T         = _mp_so_exc_ph[imode] @ pb_inv_D2              # (Nexc, Nfreq)
    return imode, -(_mp_pa_inv_D1 * T).sum(axis=0)            # (Nfreq,)

def calculate_tensor_not_vectorized_triple_resonance(ialpha, ibeta):
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    ram = _gb((Nmodes, Nfreq), complex, (Nmodes, Nfreq), complex)
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — d2: (Nmodes={Nmodes}, Nfreq={Nfreq}), d3: same — total {ram:.3f} GB')

    t0 = time.perf_counter()
    M_alfa_beta = np.zeros((Nmodes, Nmodes, Nfreq), dtype=complex)

    for imode in range(Nmodes):
        for jmode in range(Nmodes):
            print(f'mode {imode+1}/{Nmodes}, (alpha = {ialpha+1}, beta = {ibeta+1}). Not vectorized.')

            # d2: diagonal exciton sum
            acc = np.zeros(Nfreq, dtype=complex)
            for iA in range(Nexc_Q0):
                for iB in range(Nexc_Qq):
                    for iC in range(Nexc_Q0):
                        # <0|r_alpha|A><A|dH/dr_imode|B><B|dH†/dr_jmode|C><C|r_beta|0>
                        num = pa[iA] * pb[iC].conjugate() * exc_ph[imode, iA, iB] * exc_ph_dag[jmode, iB, iC]
                        denom = (Ex - exc_energies_Q0[iA] + 1j*gamma) * (Ex - freqs_eV[imode] - exc_energies_Qq[iB] + 1j*gamma) * (Ex - freqs_eV[imode] - freqs_eV[jmode] - exc_energies_Q0[iC] + 1j*gamma)
                        acc += num / denom
            M_alfa_beta[imode, jmode] = -acc
            print(f'            (alpha={ialpha+1}, beta={ibeta+1}), imode {imode+1}/{Nmodes}, jmode {jmode+1}/{Nmodes} done.')

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return M_alfa_beta


def calculate_tensor_vectorize_over_excitons_triple_resonance(ialpha, ibeta):
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    ram = _gb((Nexc, Nfreq), complex,        # D1, inv_D1, pa_inv_D1
              (Nexc, Nfreq), complex,
              (Nexc, Nfreq), complex,
              (Nexc, Nfreq), complex,        # inv_D2, inv_D3, pb_inv_D3, T2, T3, T4 (per imode,jmode)
              (Nexc, Nfreq), complex,
              (Nexc, Nfreq), complex,
              (Nexc, Nfreq), complex,
              (Nexc, Nfreq), complex,
              (Nexc, Nfreq), complex,
              (Nmodes, Nmodes, Nfreq), complex)  # M
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — D1, inv_D1, pa_inv_D1, inv_D2, inv_D3, '
          f'pb_inv_D3, T2, T3, T4, M — total {ram:.3f} GB')

    t0 = time.perf_counter()
    M = np.zeros((Nmodes, Nmodes, Nfreq), dtype=complex)

    # D1[A, f] = Ex[f] - E_A^{Q=0} + i*gamma  (Nexc_Q0, Nfreq) — mode-independent
    D1 = Ex[np.newaxis, :] - exc_energies_Q0[:, np.newaxis] + 1j * gamma
    inv_D1 = 1.0 / D1                                         # (Nexc_Q0, Nfreq)
    pa_inv_D1 = pa[:, np.newaxis] * inv_D1                    # (Nexc_Q0, Nfreq)

    # D2_base[B, f] = Ex[f] - E_B^{Q=q} + i*gamma  (Nexc_Qq, Nfreq) — for intermediate B state
    D2_base = Ex[np.newaxis, :] - exc_energies_Qq[:, np.newaxis] + 1j * gamma

    for imode in range(Nmodes):
        inv_D2 = 1.0 / (D2_base - freqs_eV[imode])           # (Nexc_Qq, Nfreq)

        for jmode in range(Nmodes):
            print(f'imode {imode+1}/{Nmodes}, jmode {jmode+1}/{Nmodes}, '
                  f'(alpha={ialpha+1}, beta={ibeta+1}). Vectorized over excitons.')

            inv_D3 = 1.0 / (D1 - freqs_eV[imode] - freqs_eV[jmode])  # (Nexc_Q0, Nfreq)

            # T1[C, f] = pb[C]* * inv_D3[C, f]
            pb_inv_D3 = pb.conjugate()[:, np.newaxis] * inv_D3        # (Nexc_Q0, Nfreq)

            # T2[B, f] = sum_C g†[jmode, B, C] * T1[C, f]  (second phonon vertex: g_q†)
            T2 = exc_ph_dag[jmode] @ pb_inv_D3                        # (Nexc_Qq, Nfreq)

            # T3[B, f] = T2[B, f] * inv_D2[B, f]
            T3 = T2 * inv_D2                                          # (Nexc_Qq, Nfreq)

            # T4[A, f] = sum_B g[imode, A, B] * T3[B, f]  (first phonon vertex: g_q)
            T4 = exc_ph[imode] @ T3                                   # (Nexc_Q0, Nfreq)

            # M[imode, jmode, f] = -sum_A pa[A] * T4[A, f] * inv_D1[A, f]
            M[imode, jmode] = -(pa_inv_D1 * T4).sum(axis=0)

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return M


def calculate_tensor_vectorized_over_modes_and_excitons_triple_resonance(ialpha, ibeta):
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    ram = _gb((Nexc, Nfreq),               complex,   # D1
              (Nexc, Nfreq),               complex,   # inv_D1
              (Nexc, Nfreq),               complex,   # pa_inv_D1
              (Nmodes, Nexc, Nfreq),       complex,   # inv_D2
              (Nmodes, Nmodes, Nexc, Nfreq), complex, # inv_D3
              (Nmodes, Nmodes, Nexc, Nfreq), complex, # pb_inv_D3
              (Nmodes, Nmodes, Nexc, Nfreq), complex, # T2
              (Nmodes, Nmodes, Nexc, Nfreq), complex, # T3
              (Nmodes, Nmodes, Nexc, Nfreq), complex, # T4
              (Nmodes, Nmodes, Nfreq),     complex)   # M
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — D1, inv_D1, pa_inv_D1, inv_D2, inv_D3, '
          f'pb_inv_D3, T2, T3, T4, M — total {ram:.3f} GB')

    t0 = time.perf_counter()
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) - vectorized over modes and excitons')

    # D1[A, f] = Ex[f] - E_A^{Q=0} + i*gamma             (Nexc_Q0, Nfreq)
    D1 = Ex[np.newaxis, :] - exc_energies_Q0[:, np.newaxis] + 1j * gamma
    inv_D1 = 1.0 / D1                                          # (Nexc_Q0, Nfreq)
    pa_inv_D1 = pa[:, np.newaxis] * inv_D1                     # (Nexc_Q0, Nfreq)

    # D2_base[B, f] = Ex[f] - E_B^{Q=q} + i*gamma       (Nexc_Qq, Nfreq)
    D2_base = Ex[np.newaxis, :] - exc_energies_Qq[:, np.newaxis] + 1j * gamma

    # inv_D2[i, B, f] = 1/(D2_base[B,f] - freqs[i])     (Nmodes, Nexc_Qq, Nfreq)
    inv_D2 = 1.0 / (D2_base[np.newaxis] - freqs_eV[:, np.newaxis, np.newaxis])

    # inv_D3[i, j, C, f] = 1/(D1[C,f] - freqs[i] - freqs[j])  (Nmodes, Nmodes, Nexc_Q0, Nfreq)
    sum_freqs = freqs_eV[:, np.newaxis] + freqs_eV[np.newaxis, :]    # (Nmodes, Nmodes)
    inv_D3 = 1.0 / (D1[np.newaxis, np.newaxis] - sum_freqs[:, :, np.newaxis, np.newaxis])

    # T1[i, j, C, f] = pb[C]* * inv_D3[i, j, C, f]
    pb_inv_D3 = pb.conjugate()[np.newaxis, np.newaxis, :, np.newaxis] * inv_D3  # (Nmodes, Nmodes, Nexc_Q0, Nfreq)

    # T2[i, j, B, f] = sum_C g†[j, B, C] * T1[i, j, C, f]  (second vertex: g_q†)
    # exc_ph_dag[np.newaxis]: (1, Nmodes, Nexc_Qq, Nexc_Q0); matmul contracts over C (Nexc_Q0)
    T2 = np.matmul(exc_ph_dag[np.newaxis], pb_inv_D3)         # (Nmodes, Nmodes, Nexc_Qq, Nfreq)

    # T3[i, j, B, f] = T2[i,j,B,f] * inv_D2[i,B,f]
    T3 = T2 * inv_D2[:, np.newaxis]                           # (Nmodes, Nmodes, Nexc_Qq, Nfreq)

    # T4[i, j, A, f] = sum_B g[i, A, B] * T3[i, j, B, f]  (first vertex: g_q)
    # exc_ph[:, np.newaxis]: (Nmodes, 1, Nexc_Q0, Nexc_Qq); matmul contracts over B (Nexc_Qq)
    T4 = np.matmul(exc_ph[:, np.newaxis], T3)                 # (Nmodes, Nmodes, Nexc_Q0, Nfreq)

    # M[i, j, f] = -sum_A pa[A] * T4[i,j,A,f] * inv_D1[A,f]
    M = -(pa_inv_D1[np.newaxis, np.newaxis] * T4).sum(axis=2) # (Nmodes, Nmodes, Nfreq)

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return M

def calculate_tensor_not_vectorized_double_resonance(ialpha, ibeta):
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    ram = _gb((Nmodes, Nfreq), complex, (Nmodes, Nfreq), complex)
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — d2: (Nmodes={Nmodes}, Nfreq={Nfreq}), d3: same — total {ram:.3f} GB')

    t0 = time.perf_counter()
    M_alfa_beta = np.zeros((Nmodes, Nfreq), dtype=complex)

    for imode in range(Nmodes):
        print(f'mode {imode+1}/{Nmodes}, (alpha = {ialpha+1}, beta = {ibeta+1}). Not vectorized.')

        # d2: diagonal exciton sum
        acc = np.zeros(Nfreq, dtype=complex)
        for iA in range(Nexc):
            for iB in range(Nexc):
                # <0|r_alpha|A><A|d^2H/dr^2_imode|B><C|r_beta|0>
                num = pa[iA] * pb[iB].conjugate() * second_order_exc_ph[imode, iA, iB]
                denom = (Ex - exc_energies[iA] + 1j*gamma) * (Ex - 2*freqs_eV[imode] - exc_energies[iB] + 1j*gamma)
                acc += num / denom
        M_alfa_beta[imode] = -acc

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return M_alfa_beta

def calculate_tensor_vectorize_over_excitons_double_resonance(ialpha, ibeta):
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    ram = _gb((Nexc, Nfreq), complex,   # D1, inv_D1, pa_inv_D1
              (Nexc, Nfreq), complex,
              (Nexc, Nfreq), complex,
              (Nexc, Nfreq), complex,   # inv_D2, pb_inv_D2, T (per mode)
              (Nexc, Nfreq), complex,
              (Nexc, Nfreq), complex,
              (Nmodes, Nfreq), complex) # M
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — D1, inv_D1, pa_inv_D1, inv_D2, pb_inv_D2, T, M — total {ram:.3f} GB')

    t0 = time.perf_counter()
    M = np.zeros((Nmodes, Nfreq), dtype=complex)

    # D1[A, f] = Ex[f] - exc_energies[A] + i*gamma  (Nexc, Nfreq) — mode-independent
    D1 = Ex[np.newaxis, :] - exc_energies[:, np.newaxis] + 1j * gamma
    inv_D1 = 1.0 / D1                                       # (Nexc, Nfreq)
    pa_inv_D1 = pa[:, np.newaxis] * inv_D1                  # (Nexc, Nfreq)

    for imode in range(Nmodes):
        print(f'mode {imode+1}/{Nmodes}, (alpha = {ialpha+1}, beta = {ibeta+1}). Vectorized over excitons, not modes.')

        inv_D2 = 1.0 / (D1 - 2*freqs_eV[imode])             # (Nexc, Nfreq)

        # T[A, f] = sum_B g2[m,A,B] * pb[B]* * inv_D2[B,f]
        pb_inv_D2 = pb.conjugate()[:, np.newaxis] * inv_D2  # (Nexc, Nfreq)
        T = second_order_exc_ph[imode] @ pb_inv_D2          # (Nexc, Nfreq)

        # M[m, f] = -sum_A pa[A] * T[A,f] * inv_D1[A,f]
        M[imode] = -(pa_inv_D1 * T).sum(axis=0)

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return M


def calculate_tensor_vectorize_over_excitons_double_resonance_parallel(ialpha, ibeta, nworkers=None):
    """
    Same math as calculate_tensor_vectorize_over_excitons_double_resonance.
    The imode loop is parallelised with multiprocessing (fork context).

    Large read-only arrays (D1, pa_inv_D1, second_order_exc_ph, …) are stored
    in module-level globals before the pool is created so that forked workers
    inherit them via copy-on-write — no per-task pickling of big arrays.
    Only the small (Nfreq,) result is sent back per mode.
    """
    global _mp_D1, _mp_pa_inv_D1, _mp_pb_conj, _mp_so_exc_ph, _mp_freqs_eV

    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    ram = _gb((Nexc, Nfreq), complex,   # D1
              (Nexc, Nfreq), complex,   # inv_D1
              (Nexc, Nfreq), complex,   # pa_inv_D1
              (Nexc, Nfreq), complex,   # inv_D2  (per worker — amortised)
              (Nexc, Nfreq), complex,   # pb_inv_D2
              (Nexc, Nfreq), complex,   # T
              (Nmodes, Nfreq), complex) # M
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — D1, inv_D1, pa_inv_D1, '
          f'inv_D2, pb_inv_D2, T, M — total {ram:.3f} GB')

    t0 = time.perf_counter()

    # Mode-independent quantities — computed once in the parent process
    D1        = Ex[np.newaxis, :] - exc_energies[:, np.newaxis] + 1j * gamma  # (Nexc, Nfreq)
    inv_D1    = 1.0 / D1                                                        # (Nexc, Nfreq)
    pa_inv_D1 = pa[:, np.newaxis] * inv_D1                                      # (Nexc, Nfreq)

    # Publish to module globals before forking — workers see them for free (COW)
    _mp_D1        = D1
    _mp_pa_inv_D1 = pa_inv_D1
    _mp_pb_conj   = pb.conjugate()
    _mp_so_exc_ph = second_order_exc_ph
    _mp_freqs_eV  = freqs_eV

    n = nworkers or os.cpu_count()
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) — multiprocess over {Nmodes} modes, {n} workers')

    M = np.zeros((Nmodes, Nfreq), dtype=complex)
    ctx = mp.get_context('fork')
    with ctx.Pool(n) as pool:
        for imode, val in pool.imap_unordered(_dbl_worker, range(Nmodes)):
            M[imode] = val
            print(f'    mode {imode+1}/{Nmodes} done')

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return M


def calculate_tensor_vectorized_over_modes_and_excitons_double_resonance(ialpha, ibeta):
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    ram = _gb((Nexc, Nfreq),         complex,   # D1
              (Nexc, Nfreq),         complex,   # inv_D1
              (Nexc, Nfreq),         complex,   # pa_inv_D1
              (Nmodes, Nexc, Nfreq), complex,   # inv_D2
              (Nmodes, Nexc, Nfreq), complex,   # pb_inv_D2
              (Nmodes, Nexc, Nfreq), complex,   # T
              (Nmodes, Nfreq),       complex)   # M
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — D1, inv_D1, pa_inv_D1, inv_D2, pb_inv_D2, T, M — total {ram:.3f} GB')

    t0 = time.perf_counter()
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) - vectorized over modes and excitons')

    # D1[A, f] = Ex[f] - exc_energies[A] + i*gamma         (Nexc, Nfreq)
    D1 = Ex[np.newaxis, :] - exc_energies[:, np.newaxis] + 1j * gamma
    inv_D1 = 1.0 / D1                                       # (Nexc, Nfreq)
    pa_inv_D1 = pa[:, np.newaxis] * inv_D1                  # (Nexc, Nfreq)

    # inv_D2[m, B, f] = 1 / (D1[B, f] - 2*freqs_eV[m])    (Nmodes, Nexc, Nfreq)
    inv_D2 = 1.0 / (D1[np.newaxis] - 2*freqs_eV[:, np.newaxis, np.newaxis])

    # T[m, A, f] = sum_B g2[m,A,B] * pb[B]* * inv_D2[m,B,f]
    pb_inv_D2 = pb.conjugate()[np.newaxis, :, np.newaxis] * inv_D2  # (Nmodes, Nexc, Nfreq)
    T = np.matmul(second_order_exc_ph, pb_inv_D2)                   # (Nmodes, Nexc, Nfreq)

    # M[m, f] = -sum_A pa[A] * T[m,A,f] * inv_D1[A,f]
    M = -(pa_inv_D1[np.newaxis] * T).sum(axis=1)                    # (Nmodes, Nfreq)

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return M


parser = argparse.ArgumentParser()
parser.add_argument('--first_order_exc_ph_file', default='1st_order_exciton_phonon_couplings.h5')
parser.add_argument('--second_order_exc_ph_file', default=None,
                    help='2nd-order exciton-phonon h5 file. If not provided, <A|d^2H/dRa dRb|B> is assumed zero.')
parser.add_argument('--dip_mom_file_b1', default='eigenvalues_b1.dat')
parser.add_argument('--dip_mom_file_b2', default='eigenvalues_b2.dat')
parser.add_argument('--dip_mom_file_b3', default='eigenvalues_b3.dat')
parser.add_argument('--dE', type=float, default=0.001, help='Energy step in eV for the Ex grid')
parser.add_argument('--gamma', type=float, default=0.01, help='Broadening parameter in eV')
parser.add_argument('--vectorized_flavor', type=int, default=2, choices=[0, 1, 2], help='0: no vectorization (triple loop), 1: vectorize over excitons only, 2: vectorize over both excitons and modes (more memory usage but faster)')
parser.add_argument('--nworkers', type=int, default=None,
                    help='Number of worker processes for double-resonance (flavor 1 only). '
                         'Default: None (serial). Set to -1 to use all CPUs.')
parser.add_argument('--test_functions', action='store_true', help='Run all three implementations on Nexc=10 and check they agree')
parser.add_argument('--freqs_file', default=None, help='File containing phonon frequencies in cm^-1 (optional if stored in --first_order_exc_ph_file)')
parser.add_argument('--limit_Nexc', type=int, default=None, help='Limit number of excitons to load (for testing)')
parser.add_argument('--finite-q', dest='finite_q', action='store_true',
                    help='Finite-q phonon mode: loads exc_A/B_energies from h5, uses g_q† for second phonon vertex.')
parser.add_argument('--output', default='susceptibility_tensors_second_order.h5',
                    help='Output h5 file (default: susceptibility_tensors_second_order.h5)')
parser.add_argument('--write_dummy', action='store_true',
                    help='Also compute and save dummy tensors (all numerators = 1, joint DOS of transitions) '
                         'to susceptibility_tensors_second_order_dummy.h5')
args = parser.parse_args()

first_order_exc_ph_file = args.first_order_exc_ph_file
second_order_exc_ph_file = args.second_order_exc_ph_file
dip_mom_file_b1 = args.dip_mom_file_b1
dip_mom_file_b2 = args.dip_mom_file_b2
dip_mom_file_b3 = args.dip_mom_file_b3
vectorized_flavor = args.vectorized_flavor
nworkers          = None if args.nworkers is None else (None if args.nworkers == -1 else args.nworkers)
test_functions = args.test_functions
freqs_file = args.freqs_file
dE = args.dE
Emin = 0.0  # Minimum excitation energy in eV
gamma = args.gamma  # eV
limit_Nexc = args.limit_Nexc # int or None
finite_q = args.finite_q
output_h5_file = args.output

flavor_desc = {0: 'no vectorization (triple loop)',
               1: 'vectorized over excitons',
               2: 'vectorized over excitons and modes (fully vectorized)'}
print('--- Options ---')
print(f'  1st_order_exc_ph_file : {first_order_exc_ph_file}')
print(f'  2nd_order_exc_ph_file : {second_order_exc_ph_file if second_order_exc_ph_file else "(not provided, assumed zero)"}')
print(f'  freqs_file        : {freqs_file if freqs_file else "(read from h5)"}')
print(f'  dip_mom_file_b1   : {dip_mom_file_b1}')
print(f'  dip_mom_file_b2   : {dip_mom_file_b2}')
print(f'  dip_mom_file_b3   : {dip_mom_file_b3}')
print(f'  dE                : {dE} eV')
print(f'  gamma             : {gamma} eV')
print(f'  vectorized_flavor : {vectorized_flavor} ({flavor_desc[vectorized_flavor]})')
print(f'  nworkers          : {args.nworkers} (double-resonance parallel processes; None=serial, -1=all CPUs)')
print(f'  test_functions    : {test_functions}')
print(f'  limit_Nexc        : {args.limit_Nexc} (None means no limit)')
print(f'  finite_q          : {finite_q}')
print(f'  output            : {output_h5_file}')
print(f'  write_dummy       : {args.write_dummy}')
print('---------------\n')

freqs_eV = None  # loaded below from h5 or --freqs_file

def _load_exc_ph_h5(fname, hermitian=True):
    """Load exciton-phonon matrix from an exc_forces/assembled h5.
    hermitian=True (Q=0): square (Nmodes, Nexc, Nexc) with Hermitian symmetry filled.
    hermitian=False (finite-q): rectangular (Nmodes, Nexc_A, Nexc_B) without symmetry fill.
    Returns (mat, ph_freqs, enc_A, enc_B)."""
    with h5py.File(fname, 'r') as hf:
        if 'forces/ph/RPA' in hf:
            pairs  = hf['exciton_pairs'][:]  # (Npairs, 2) 1-based
            forces = hf['forces/ph/RPA'][:]  # (Npairs, Nmodes)
            _Nm = forces.shape[1]
            if hermitian:
                max_exc = int(pairs.max())
                mat = np.zeros((_Nm, max_exc, max_exc), dtype=complex)
            else:
                max_A = int(pairs[:, 0].max())
                max_B = int(pairs[:, 1].max())
                mat = np.zeros((_Nm, max_A, max_B), dtype=complex)
            for k, (i, j) in enumerate(pairs):
                val = -forces[k]             # negate: F = -<A|dH|B> → <A|dH|B>
                mat[:, i-1, j-1] = val
                if hermitian and i != j:
                    mat[:, j-1, i-1] = val.conj()
            if hermitian:
                print(f'  Built {_Nm}×{mat.shape[1]}×{mat.shape[2]} matrix from {len(pairs)} pairs '
                      f'(missing pairs zero; Hermitian symmetry applied)')
            else:
                print(f'  Built {_Nm}×{mat.shape[1]}×{mat.shape[2]} matrix from {len(pairs)} pairs '
                      f'(finite-q: no Hermitian fill; rows=Q=0, cols=Q=q)')
            ph_freqs = hf['system/phonon_frequencies'][:] if 'system/phonon_frequencies' in hf else None
            enc_A = hf['system/exc_A_energies'][:] if 'system/exc_A_energies' in hf else None
            enc_B = hf['system/exc_B_energies'][:] if 'system/exc_B_energies' in hf else None
        else:
            mat = hf['rpa_offdiag'][:]
            print(f'  Loaded old-format matrix: shape {mat.shape}')
            ph_freqs = None
            enc_A = None
            enc_B = None
    return mat, ph_freqs, enc_A, enc_B

print(f'Reading 1st-order exciton-phonon data from {first_order_exc_ph_file}')
exc_ph, _freqs, _enc_A, _enc_B = _load_exc_ph_h5(first_order_exc_ph_file, hermitian=(not finite_q))
if _freqs is not None:
    freqs_eV = _freqs * rec_cm_to_eV
    print(f'  Loaded {len(freqs_eV)} phonon frequencies from h5')

# For finite-q: Q=q exciton energies for intermediate state come from the h5 file
if finite_q:
    if _enc_B is None:
        sys.exit('ERROR: system/exc_B_energies not found in h5. '
                 'Re-run excited_forces.py (it will save exc_A/B_energies).')
    exc_energies_Qq_h5 = _enc_B  # (max_B,) — filled at loaded B-exciton slots
else:
    exc_energies_Qq_h5 = None  # set later from dipole file

print('Data read successfully.')

if second_order_exc_ph_file is not None:
    print(f'Reading 2nd-order exciton-phonon data from {second_order_exc_ph_file}')
    second_order_exc_ph, _freqs2, _, _ = _load_exc_ph_h5(second_order_exc_ph_file)
    if freqs_eV is None and _freqs2 is not None:
        freqs_eV = _freqs2 * rec_cm_to_eV
        print(f'  Loaded {len(freqs_eV)} phonon frequencies from 2nd-order h5')
    print('Data read successfully.')
    if not finite_q:
        # Nexc consistency only meaningful for Q=0 (square matrices)
        _Nexc_1 = exc_ph.shape[1]
        _Nexc_2 = second_order_exc_ph.shape[1]
        if _Nexc_1 != _Nexc_2:
            _Nexc = min(_Nexc_1, _Nexc_2)
            print(f'WARNING: 1st-order Nexc={_Nexc_1} ≠ 2nd-order Nexc={_Nexc_2}; truncating both to {_Nexc}')
            exc_ph              = exc_ph[:, :_Nexc, :_Nexc]
            second_order_exc_ph = second_order_exc_ph[:, :_Nexc, :_Nexc]
else:
    print('No 2nd-order exciton-phonon file provided; <A|d^2H/dRa dRb|B> assumed zero.')
    second_order_exc_ph = np.zeros((exc_ph.shape[0], exc_ph.shape[1], exc_ph.shape[1]), dtype=complex)

if freqs_eV is None:
    if freqs_file is not None:
        freqs_eV = np.loadtxt(freqs_file) * rec_cm_to_eV
        print(f'Loaded {len(freqs_eV)} phonon frequencies from {freqs_file}')
    else:
        sys.exit(
            'ERROR: phonon frequencies not found in h5 files and --freqs_file not given.\n'
            'Use assemble_exciton_phonon_coeffs.py with exc_forces.h5 input (which stores '
            'system/phonon_frequencies), or pass --freqs_file.'
        )

print(f'Reading exciton energies from {dip_mom_file_b1}')
data_eigvals_file = np.loadtxt(dip_mom_file_b1)  # shape: (Nexciton, 4)
exc_energies_Q0 = data_eigvals_file[:, 0]  # Q=0 exciton energies in eV

# exc_energies_Qq: Q=q energies for intermediate state B
if finite_q:
    # Loaded from h5 earlier; dense array (Nexc_Qq,) with zeros for not-loaded slots
    exc_energies_Qq = exc_energies_Qq_h5
else:
    exc_energies_Qq = exc_energies_Q0  # same space for Q=0

# Keep exc_energies = Q=0 for double-resonance functions (backward compat)
exc_energies = exc_energies_Q0

# Build excitation energy grid from Q=0 energies (non-zero entries only, for finite-q)
_en_for_range = exc_energies_Q0[exc_energies_Q0 > 0]
Emin = max(np.min(_en_for_range) - 0.5, 0.0)
Emax = Emin + 4.0
Ex = np.arange(Emin, Emax, dE)  # eV

data_dip_mom_b1 = np.loadtxt(dip_mom_file_b1)  # shape: (Nexciton, 4)
data_dip_mom_b2 = np.loadtxt(dip_mom_file_b2)  # shape: (Nexciton, 4)
data_dip_mom_b3 = np.loadtxt(dip_mom_file_b3)  # shape: (Nexciton, 4)
dip_moments_b1 = data_dip_mom_b1[:, 2]  + 1j * data_dip_mom_b1[:, 3] # shape (Nexciton,)
dip_moments_b2 = data_dip_mom_b2[:, 2]  + 1j * data_dip_mom_b2[:, 3] # shape (Nexciton,)
dip_moments_b3 = data_dip_mom_b3[:, 2]  + 1j * data_dip_mom_b3[:, 3] # shape (Nexciton,)
pos_operator_b1 = 1j * dip_moments_b1 / exc_energies_Q0
pos_operator_b2 = 1j * dip_moments_b2 / exc_energies_Q0
pos_operator_b3 = 1j * dip_moments_b3 / exc_energies_Q0

# Align Q=0 matrix dimension (axis 1) with dipole moment size
_Nexc_mat = exc_ph.shape[1]
_Nexc_dip = pos_operator_b1.shape[0]
if _Nexc_dip < _Nexc_mat:
    print(f'Warning: dipole moments have {_Nexc_dip} Q=0 excitons < {_Nexc_mat} in coupling data; truncating.')
    exc_ph              = exc_ph[:, :_Nexc_dip, :]
    second_order_exc_ph = second_order_exc_ph[:, :_Nexc_dip, :_Nexc_dip]
elif _Nexc_dip > _Nexc_mat:
    print(f'Warning: dipole moments have {_Nexc_dip} Q=0 excitons > {_Nexc_mat}; truncating dipole moments.')
    pos_operator_b1  = pos_operator_b1[:_Nexc_mat]
    pos_operator_b2  = pos_operator_b2[:_Nexc_mat]
    pos_operator_b3  = pos_operator_b3[:_Nexc_mat]
    exc_energies_Q0  = exc_energies_Q0[:_Nexc_mat]
    exc_energies     = exc_energies_Q0

# Align Q=q matrix dimension (axis 2) with exc_energies_Qq
_Nexc_Qq_mat = exc_ph.shape[2]
_Nexc_Qq_en  = len(exc_energies_Qq)
if _Nexc_Qq_en < _Nexc_Qq_mat:
    print(f'Warning: exc_energies_Qq length {_Nexc_Qq_en} < matrix Qq dim {_Nexc_Qq_mat}; truncating matrix.')
    exc_ph = exc_ph[:, :, :_Nexc_Qq_en]
elif _Nexc_Qq_en > _Nexc_Qq_mat:
    exc_energies_Qq = exc_energies_Qq[:_Nexc_Qq_mat]

# Conjugate transpose for second phonon vertex (g_q†)
# At Q=0: exc_ph is Hermitian, so exc_ph_dag == exc_ph
exc_ph_dag = exc_ph.conj().transpose(0, 2, 1)  # (Nmodes, Nexc_Qq, Nexc_Q0)

pos_operator_list = [pos_operator_b1, pos_operator_b2, pos_operator_b3]

print('Exciton energies read successfully.')

print('Calculating susceptibility tensors')
Nmodes  = exc_ph.shape[0]
Nexc_Q0 = exc_ph.shape[1]
Nexc_Qq = exc_ph.shape[2]
Nexc    = Nexc_Q0  # backward compat for double-resonance functions
Nfreq   = Ex.shape[0]

if test_functions:
    print('Test mode: truncating to Nexc=10')
    Nexc_Q0 = min(10, exc_ph.shape[1])
    Nexc_Qq = min(10, exc_ph.shape[2])
    exc_ph              = exc_ph[:, :Nexc_Q0, :Nexc_Qq]
    exc_ph_dag          = exc_ph.conj().transpose(0, 2, 1)
    exc_energies_Q0     = exc_energies_Q0[:Nexc_Q0]
    exc_energies_Qq     = exc_energies_Qq[:Nexc_Qq]
    exc_energies        = exc_energies_Q0
    Nexc                = Nexc_Q0
    pos_operator_list   = [p[:Nexc_Q0] for p in pos_operator_list]
    second_order_exc_ph = second_order_exc_ph[:, :Nexc_Q0, :Nexc_Q0]

    all_passed = True

    print('\n--- Testing triple-resonance implementations ---')
    results_triple = {}
    for flavor, fn in [(0, calculate_tensor_not_vectorized_triple_resonance),
                       (1, calculate_tensor_vectorize_over_excitons_triple_resonance),
                       (2, calculate_tensor_vectorized_over_modes_and_excitons_triple_resonance)]:
        M_all = np.zeros((3, 3, Nmodes, Nmodes, Nfreq), dtype=complex)
        for ialpha in range(3):
            for ibeta in range(3):
                M_all[ialpha, ibeta] = fn(ialpha, ibeta)
        results_triple[flavor] = M_all

    ref = results_triple[0]
    for flavor in [1, 2]:
        err = np.max(np.abs(results_triple[flavor] - ref))
        status = 'PASS' if err < 1e-10 else 'FAIL'
        print(f'  triple-resonance flavor {flavor} vs flavor 0 — {status} (max|diff|={err:.2e})')
        if status == 'FAIL':
            all_passed = False

    print('\n--- Testing double-resonance implementations ---')
    results_double = {}
    for flavor, fn in [(0, calculate_tensor_not_vectorized_double_resonance),
                       (1, calculate_tensor_vectorize_over_excitons_double_resonance),
                       (2, calculate_tensor_vectorized_over_modes_and_excitons_double_resonance)]:
        M_all = np.zeros((3, 3, Nmodes, Nfreq), dtype=complex)
        for ialpha in range(3):
            for ibeta in range(3):
                M_all[ialpha, ibeta] = fn(ialpha, ibeta)
        results_double[flavor] = M_all

    ref = results_double[0]
    for flavor in [1, 2]:
        err = np.max(np.abs(results_double[flavor] - ref))
        status = 'PASS' if err < 1e-10 else 'FAIL'
        print(f'  double-resonance flavor {flavor} vs flavor 0 — {status} (max|diff|={err:.2e})')
        if status == 'FAIL':
            all_passed = False

    print('\nAll tests passed.' if all_passed else '\nSome tests FAILED.')
    raise SystemExit(0)

if limit_Nexc is not None:
    print(f'Limiting to Nexc_Q0={min(limit_Nexc, Nexc_Q0)}, Nexc_Qq={min(limit_Nexc, Nexc_Qq)} as specified by --limit_Nexc')
    Nexc_Q0             = min(limit_Nexc, Nexc_Q0)
    Nexc_Qq             = min(limit_Nexc, Nexc_Qq)
    Nexc                = Nexc_Q0
    exc_ph              = exc_ph[:, :Nexc_Q0, :Nexc_Qq]
    exc_ph_dag          = exc_ph.conj().transpose(0, 2, 1)
    second_order_exc_ph = second_order_exc_ph[:, :Nexc_Q0, :Nexc_Q0]
    exc_energies_Q0     = exc_energies_Q0[:Nexc_Q0]
    exc_energies_Qq     = exc_energies_Qq[:Nexc_Qq]
    exc_energies        = exc_energies_Q0
    pos_operator_list   = [p[:Nexc_Q0] for p in pos_operator_list]

alpha_tensor_triple_res = np.zeros((3, 3, Nmodes, Nmodes, Nfreq), dtype=complex)
alpha_tensor_double_res = np.zeros((3, 3, Nmodes, Nfreq), dtype=complex)

for ialpha in range(3):
    for ibeta in range(3):
        if vectorized_flavor == 2:
            alpha_tensor_triple_res[ialpha, ibeta] = calculate_tensor_vectorized_over_modes_and_excitons_triple_resonance(ialpha, ibeta)
            alpha_tensor_double_res[ialpha, ibeta] = calculate_tensor_vectorized_over_modes_and_excitons_double_resonance(ialpha, ibeta)
        elif vectorized_flavor == 1:
            alpha_tensor_triple_res[ialpha, ibeta] = calculate_tensor_vectorize_over_excitons_triple_resonance(ialpha, ibeta)
            if nworkers is not None:
                alpha_tensor_double_res[ialpha, ibeta] = calculate_tensor_vectorize_over_excitons_double_resonance_parallel(ialpha, ibeta, nworkers=nworkers)
            else:
                alpha_tensor_double_res[ialpha, ibeta] = calculate_tensor_vectorize_over_excitons_double_resonance(ialpha, ibeta)
        else:
            alpha_tensor_triple_res[ialpha, ibeta] = calculate_tensor_not_vectorized_triple_resonance(ialpha, ibeta)
            alpha_tensor_double_res[ialpha, ibeta] = calculate_tensor_not_vectorized_double_resonance(ialpha, ibeta)

# save results to h5 file
with h5py.File(output_h5_file, 'w') as hf:
    hf.create_dataset('excitation_energies',           data=Ex)                       # (Nfreq,)
    hf.create_dataset('alpha_tensor_triple_resonance', data=alpha_tensor_triple_res)  # (3, 3, Nmodes, Nmodes, Nfreq)
    hf.create_dataset('alpha_tensor_double_resonance', data=alpha_tensor_double_res)  # (3, 3, Nmodes, Nfreq)
    hf.create_dataset('phonon_frequencies_cm', data=freqs_eV / rec_cm_to_eV)
print(f'Saved susceptibility tensors to {output_h5_file}')

# --- Dummy susceptibility tensors (numerator = 1, joint DOS of transitions) ---
if args.write_dummy:
    print('\nComputing dummy susceptibility tensors (numerator = 1)...')
    t_dummy = time.perf_counter()

    # D1[s, f] = Ex[f] - E_s + iγ  — same denominators as the main computation
    D1_mat = Ex[np.newaxis, :] - exc_energies[:, np.newaxis] + 1j * gamma  # (Nexc, Nfreq)
    inv_D1 = 1.0 / D1_mat                                                   # (Nexc, Nfreq)
    G0     = inv_D1.sum(axis=0)                                             # (Nfreq,)

    # G_nu[m, f] = Σ_s 1/(D1[s,f] - ω_m)                 (Nmodes, Nfreq)
    G_nu = (1.0 / (D1_mat[np.newaxis]
                   - freqs_eV[:, np.newaxis, np.newaxis])).sum(axis=1)

    # --- Triple resonance dummy: -G(E) · G(E-ω_i) · G(E-ω_i-ω_j) ---
    #   All three sums over excitons A, B, C are independent → fully factorises.
    #   Loop over imode to keep peak allocation at (Nmodes, Nexc, Nfreq) per step.
    print('  Triple resonance dummy...')
    M_triple_dummy = np.zeros((Nmodes, Nmodes, Nfreq), dtype=complex)
    for imode in range(Nmodes):
        # G_ij[j, f] = Σ_s 1/(D1[s,f] - ω_i - ω_j)      (Nmodes, Nfreq)
        G_ij = (1.0 / (D1_mat[np.newaxis]
                        - (freqs_eV[imode] + freqs_eV[:, np.newaxis, np.newaxis])
                       )).sum(axis=1)
        M_triple_dummy[imode] = -G0[np.newaxis] * G_nu[imode][np.newaxis] * G_ij

    # --- Double resonance dummy: -G(E) · G(E - 2ω_m) ---
    #   Independent sums over A and B → fully factorises.
    print('  Double resonance dummy...')
    G_2nu = (1.0 / (D1_mat[np.newaxis]
                    - 2.0 * freqs_eV[:, np.newaxis, np.newaxis])).sum(axis=1)  # (Nmodes, Nfreq)
    M_double_dummy = -G0[np.newaxis] * G_2nu                                    # (Nmodes, Nfreq)

    # Broadcast to (3, 3, ...) — polarisation-independent by construction
    alpha_triple_dummy = np.tile(M_triple_dummy[np.newaxis, np.newaxis], (3, 3, 1, 1, 1))
    alpha_double_dummy = np.tile(M_double_dummy[np.newaxis, np.newaxis], (3, 3, 1, 1))

    dummy_h5_file = 'susceptibility_tensors_second_order_dummy.h5'
    with h5py.File(dummy_h5_file, 'w') as hf:
        hf.create_dataset('excitation_energies',           data=Ex)
        hf.create_dataset('alpha_tensor_triple_resonance', data=alpha_triple_dummy)  # (3, 3, Nmodes, Nmodes, Nfreq)
        hf.create_dataset('alpha_tensor_double_resonance', data=alpha_double_dummy)  # (3, 3, Nmodes, Nfreq)
    print(f'Saved dummy susceptibility tensors to {dummy_h5_file}  '
          f'({time.perf_counter() - t_dummy:.3f} s)')