
import argparse
import os
import time
import numpy as np
import h5py
import multiprocessing as mp

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

def _gb(*shapes_and_dtypes):
    """Sum of array sizes in GB. Args: alternating (shape_tuple, dtype) pairs."""
    total = 0
    for shape, dtype in zip(shapes_and_dtypes[::2], shapes_and_dtypes[1::2]):
        total += np.prod(shape) * np.dtype(dtype).itemsize
    return total / 1024**3

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
            for iA in range(Nexc):
                for iB in range(Nexc):
                    for iC in range(Nexc):
                        # <0|r_alpha|A><A|dH/dr_imode|B><B|dH/dr_imode|C><C|r_beta|0>
                        num = pa[iA] * pb[iC].conjugate() * exc_ph[imode, iA, iB] * exc_ph[jmode, iB, iC]
                        denom = (Ex - exc_energies[iA] + 1j*gamma) * (Ex - freqs_eV[imode] - exc_energies[iB] + 1j*gamma) * (Ex - freqs_eV[imode] - freqs_eV[jmode] - exc_energies[iC] + 1j*gamma)
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

    # D1[A, f] = Ex[f] - exc_energies[A] + i*gamma  (Nexc, Nfreq) — mode-independent
    D1 = Ex[np.newaxis, :] - exc_energies[:, np.newaxis] + 1j * gamma
    inv_D1 = 1.0 / D1                                         # (Nexc, Nfreq)
    pa_inv_D1 = pa[:, np.newaxis] * inv_D1                    # (Nexc, Nfreq)

    for imode in range(Nmodes):
        inv_D2 = 1.0 / (D1 - freqs_eV[imode])                # (Nexc, Nfreq)

        for jmode in range(Nmodes):
            print(f'imode {imode+1}/{Nmodes}, jmode {jmode+1}/{Nmodes}, '
                  f'(alpha={ialpha+1}, beta={ibeta+1}). Vectorized over excitons.')

            inv_D3 = 1.0 / (D1 - freqs_eV[imode] - freqs_eV[jmode])  # (Nexc, Nfreq)

            # T1[C, f] = pb[C]* * inv_D3[C, f]
            pb_inv_D3 = pb.conjugate()[:, np.newaxis] * inv_D3        # (Nexc, Nfreq)

            # T2[B, f] = sum_C g[jmode, B, C] * T1[C, f]
            T2 = exc_ph[jmode] @ pb_inv_D3                            # (Nexc, Nfreq)

            # T3[B, f] = T2[B, f] * inv_D2[B, f]
            T3 = T2 * inv_D2                                          # (Nexc, Nfreq)

            # T4[A, f] = sum_B g[imode, A, B] * T3[B, f]
            T4 = exc_ph[imode] @ T3                                   # (Nexc, Nfreq)

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

    # D1[A, f] = Ex[f] - exc_energies[A] + i*gamma             (Nexc, Nfreq)
    D1 = Ex[np.newaxis, :] - exc_energies[:, np.newaxis] + 1j * gamma
    inv_D1 = 1.0 / D1                                          # (Nexc, Nfreq)
    pa_inv_D1 = pa[:, np.newaxis] * inv_D1                     # (Nexc, Nfreq)

    # inv_D2[i, B, f] = 1/(D1[B,f] - freqs[i])                (Nmodes, Nexc, Nfreq)
    inv_D2 = 1.0 / (D1[np.newaxis] - freqs_eV[:, np.newaxis, np.newaxis])

    # inv_D3[i, j, C, f] = 1/(D1[C,f] - freqs[i] - freqs[j])  (Nmodes, Nmodes, Nexc, Nfreq)
    sum_freqs = freqs_eV[:, np.newaxis] + freqs_eV[np.newaxis, :]    # (Nmodes, Nmodes)
    inv_D3 = 1.0 / (D1[np.newaxis, np.newaxis] - sum_freqs[:, :, np.newaxis, np.newaxis])

    # T1[i, j, C, f] = pb[C]* * inv_D3[i, j, C, f]
    pb_inv_D3 = pb.conjugate()[np.newaxis, np.newaxis, :, np.newaxis] * inv_D3  # (Nmodes, Nmodes, Nexc, Nfreq)

    # T2[i, j, B, f] = sum_C g[j, B, C] * T1[i, j, C, f]
    # exc_ph[np.newaxis]: (1, Nmodes, Nexc, Nexc) broadcasts over i; matmul contracts over C
    T2 = np.matmul(exc_ph[np.newaxis], pb_inv_D3)             # (Nmodes, Nmodes, Nexc, Nfreq)

    # T3[i, j, B, f] = T2[i,j,B,f] * inv_D2[i,B,f]
    T3 = T2 * inv_D2[:, np.newaxis]                           # (Nmodes, Nmodes, Nexc, Nfreq)

    # T4[i, j, A, f] = sum_B g[i, A, B] * T3[i, j, B, f]
    # exc_ph[:, np.newaxis]: (Nmodes, 1, Nexc, Nexc) broadcasts over j; matmul contracts over B
    T4 = np.matmul(exc_ph[:, np.newaxis], T3)                 # (Nmodes, Nmodes, Nexc, Nfreq)

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
parser.add_argument('--second_order_exc_ph_file', default='2nd_order_exciton_phonon_couplings.h5')
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
parser.add_argument('--freqs_file', default='freqs.dat', help='File containing phonon frequencies in cm^-1')
parser.add_argument('--limit_Nexc', type=int, default=None, help='Limit number of excitons to load (for testing)')
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

flavor_desc = {0: 'no vectorization (triple loop)',
               1: 'vectorized over excitons',
               2: 'vectorized over excitons and modes (fully vectorized)'}
print('--- Options ---')
print(f'  1st_order_exc_ph_file : {first_order_exc_ph_file}')
print(f'  2nd_order_exc_ph_file : {second_order_exc_ph_file}')
print(f'  freqs_file        : {freqs_file}')
print(f'  dip_mom_file_b1   : {dip_mom_file_b1}')
print(f'  dip_mom_file_b2   : {dip_mom_file_b2}')
print(f'  dip_mom_file_b3   : {dip_mom_file_b3}')
print(f'  dE                : {dE} eV')
print(f'  gamma             : {gamma} eV')
print(f'  vectorized_flavor : {vectorized_flavor} ({flavor_desc[vectorized_flavor]})')
print(f'  nworkers          : {args.nworkers} (double-resonance parallel processes; None=serial, -1=all CPUs)')
print(f'  test_functions    : {test_functions}')
print(f'  limit_Nexc        : {args.limit_Nexc} (None means no limit)')
print('---------------\n')

rec_cm_to_eV = 1.239841984e-4  # Conversion factor from cm^-1 to eV
freqs_rec_cm = np.loadtxt(freqs_file)  # Load phonon frequencies in cm^-1
freqs_eV = freqs_rec_cm * rec_cm_to_eV  # Convert frequencies. shape (Nmodes,)

# reading file produced by assemble_exciton_phonon_coeffs.py
print(f'Reading data from {exc_ph_file}')
with h5py.File(exc_ph_file, 'r') as hf:
    # rpa_diag_data = hf['rpa_diag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
    exc_ph = hf['rpa_offdiag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
print('Data read successfully.')

print(f'Reading data from {second_order_exc_ph_file}')
with h5py.File(second_order_exc_ph_file, 'r') as hf:
    second_order_exc_ph = hf['rpa_offdiag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
print('Data read successfully.')

print(f'Reading exciton energies from {dip_mom_file_b1}')
data_eigvals_file = np.loadtxt(dip_mom_file_b1)  # shape: (Nexciton, 4)
exc_energies = data_eigvals_file[:, 0]  # in eV

Emin = max(np.min(exc_energies) - 0.5, 0.0)  # eV, set Emin to 0.5 eV below the lowest exciton energy, but not below 0
Emax = np.max(exc_energies) + 0.5  # eV
Emax = Emin + 4.0 
Ex = np.arange(Emin, Emax, dE)  # eV

data_dip_mom_b1 = np.loadtxt(dip_mom_file_b1)  # shape: (Nexciton, 4)
data_dip_mom_b2 = np.loadtxt(dip_mom_file_b2)  # shape: (Nexciton, 4)
data_dip_mom_b3 = np.loadtxt(dip_mom_file_b3)  # shape: (Nexciton, 4)
dip_moments_b1 = data_dip_mom_b1[:, 2]  + 1j * data_dip_mom_b1[:, 3] # shape (Nexciton,)
dip_moments_b2 = data_dip_mom_b2[:, 2]  + 1j * data_dip_mom_b2[:, 3] # shape (Nexciton,)
dip_moments_b3 = data_dip_mom_b3[:, 2]  + 1j * data_dip_mom_b3[:, 3] # shape (Nexciton,)
pos_operator_b1 = 1j * dip_moments_b1 / exc_energies  # <0|r|i> = i * <0|p|i> / E_ixc
pos_operator_b2 = 1j * dip_moments_b2 / exc_energies  # shape (Nexciton,)
pos_operator_b3 = 1j * dip_moments_b3 / exc_energies  # shape (Nexciton,)

if pos_operator_b1.shape[0] < exc_ph.shape[1]:
    print(f'Warning: number of excitons in dipole moments ({pos_operator_b1.shape[0]}) is less than number of excitons in exciton-phonon coupling data ({exc_ph.shape[1]}). Truncating to {pos_operator_b1.shape[0]} excitons.')
    exc_ph = exc_ph[:, :pos_operator_b1.shape[0], :pos_operator_b1.shape[0]]
elif pos_operator_b1.shape[0] > exc_ph.shape[1]:
    print(f'Warning: number of excitons in dipole moments ({pos_operator_b1.shape[0]}) is greater than number of excitons in exciton-phonon coupling data ({exc_ph.shape[1]}). Truncating to {exc_ph.shape[1]} excitons.')
    pos_operator_b1 = pos_operator_b1[:exc_ph.shape[1]]
    pos_operator_b2 = pos_operator_b2[:exc_ph.shape[1]]
    pos_operator_b3 = pos_operator_b3[:exc_ph.shape[1]]
    exc_energies = exc_energies[:exc_ph.shape[1]]

pos_operator_list = [pos_operator_b1, pos_operator_b2, pos_operator_b3]

print('Exciton energies read successfully.')

print('Calculating susceptibility tensors')
Nmodes = exc_ph.shape[0]
Nexc = exc_ph.shape[1]
Nfreq = Ex.shape[0]  # number of energies for incident light

if test_functions:
    print('Test mode: truncating to Nexc=10')
    Nexc = 10
    exc_ph = exc_ph[:, :Nexc, :Nexc]
    exc_energies = exc_energies[:Nexc]
    pos_operator_list = [p[:Nexc] for p in pos_operator_list]

    second_order_exc_ph = second_order_exc_ph[:, :Nexc, :Nexc]

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
    print(f'Limiting to Nexc={limit_Nexc} excitons as specified by --limit_Nexc')
    exc_ph = exc_ph[:, :limit_Nexc, :limit_Nexc]
    second_order_exc_ph = second_order_exc_ph[:, :limit_Nexc, :limit_Nexc]
    exc_energies = exc_energies[:limit_Nexc]
    pos_operator_list = [p[:limit_Nexc] for p in pos_operator_list]

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
output_h5_file = 'susceptibility_tensors_second_order.h5'
with h5py.File(output_h5_file, 'w') as hf:
    hf.create_dataset('excitation_energies',           data=Ex)                       # (Nfreq,)
    hf.create_dataset('alpha_tensor_triple_resonance', data=alpha_tensor_triple_res)  # (3, 3, Nmodes, Nmodes, Nfreq)
    hf.create_dataset('alpha_tensor_double_resonance', data=alpha_tensor_double_res)  # (3, 3, Nmodes, Nfreq)
print(f'Saved susceptibility tensors to {output_h5_file}')