
import argparse
import time
import numpy as np
import h5py

def _gb(*shapes_and_dtypes):
    """Sum of array sizes in GB. Args: alternating (shape_tuple, dtype) pairs."""
    total = 0
    for shape, dtype in zip(shapes_and_dtypes[::2], shapes_and_dtypes[1::2]):
        total += np.prod(shape) * np.dtype(dtype).itemsize
    return total / 1024**3

def calculate_tensor_not_vectorized(ialpha, ibeta):
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    ram = _gb((Nmodes, Nfreq), complex, (Nmodes, Nfreq), complex)
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — d2: (Nmodes={Nmodes}, Nfreq={Nfreq}), d3: same — total {ram:.3f} GB')

    t0 = time.perf_counter()
    d2 = np.zeros((Nmodes, Nfreq), dtype=complex)
    d3 = np.zeros((Nmodes, Nfreq), dtype=complex)

    for imode in range(Nmodes):
        print(f'mode {imode+1}/{Nmodes}, (alpha = {ialpha+1}, beta = {ibeta+1}). Not vectorized.')

        # d2: diagonal exciton sum
        acc = np.zeros(Nfreq, dtype=complex)
        for iexc in range(Nexc):
            # <0|r_alpha|S><S|dH/dr_imode|S><S|r_beta|0>
            num = pa[iexc] * pb[iexc].conjugate() * exc_ph[imode, iexc, iexc]
            denom = (Ex - exc_energies[iexc] + 1j*gamma) * (Ex + freqs_eV[imode] - exc_energies[iexc] + 1j*gamma)
            acc += num / denom
        d2[imode] = -acc

        # d2 + d3: full (Nexc x Nexc) sum
        acc = np.zeros(Nfreq, dtype=complex)
        for iexc1 in range(Nexc):
            for iexc2 in range(Nexc):
                # <0|r_alpha|S><S|dH/dr_imode|S'><S'|r_beta|0>
                num = pa[iexc1] * pb[iexc2].conjugate() * exc_ph[imode, iexc1, iexc2]
                denom = (Ex - exc_energies[iexc1] + 1j*gamma) * (Ex + freqs_eV[imode] - exc_energies[iexc2] + 1j*gamma)
                acc += num / denom
        d3[imode] = -acc

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return d2, d3


def calculate_tensor_vectorize_over_excitons(ialpha, ibeta):
    pa = pos_operator_list[ialpha]
    pb = pos_operator_list[ibeta]

    ram = _gb((Nmodes, Nfreq), complex, (Nmodes, Nfreq), complex,
              (Nexc, Nexc),   complex, (Nexc, Nfreq),   complex,
              (Nexc, Nfreq),  complex, (Nexc, Nfreq),   complex)
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — d2, d3, P_outer, D1, inv_D2, temp (per mode) — total {ram:.3f} GB')

    t0 = time.perf_counter()
    d2 = np.zeros((Nmodes, Nfreq), dtype=complex)
    d3 = np.zeros((Nmodes, Nfreq), dtype=complex)

    # P_outer[s1, s2] = pa[s1] * pb[s2].conj  (Nexc, Nexc)
    P_outer = pa[:, np.newaxis] * pb[np.newaxis, :].conjugate()

    # D1[s, f] = Ex[f] - exc_energies[s] + i*gamma  (Nexc, Nfreq) — mode-independent
    D1 = Ex[np.newaxis, :] - exc_energies[:, np.newaxis] + 1j * gamma

    for imode in range(Nmodes):
        print(f'mode {imode+1}/{Nmodes}, (alpha = {ialpha+1}, beta = {ibeta+1}). Vectorized over excitons, not modes.')

        # D2[s, f] = D1[s, f] + freqs_eV[imode]  (Nexc, Nfreq)
        inv_D2 = 1.0 / (D1 + freqs_eV[imode])

        # d2: sum_s P_diag[s] * exc_ph[m,s,s] * inv_D1[s,f] * inv_D2[s,f]
        num_d2 = np.diag(P_outer) * np.diag(exc_ph[imode])  # (Nexc,)
        d2[imode] = -(num_d2[:, np.newaxis] / D1 * inv_D2).sum(axis=0)

        # d2 + d3: factored matmul avoids allocating (Nexc, Nexc, Nfreq)
        #   temp[s1, f] = sum_s2 P_outer[s1,s2] * exc_ph[m,s1,s2] * inv_D2[s2,f]
        #   d3[m, f]    = -sum_s1 temp[s1, f] / D1[s1, f]
        num_full = P_outer * exc_ph[imode]           # (Nexc, Nexc)
        temp = num_full @ inv_D2                     # (Nexc, Nfreq)
        d3[imode] = -(temp / D1).sum(axis=0)

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return d2, d3


def calculate_tensor_vectorized_over_modes_and_excitons(ialpha, ibeta):
    ram = _gb((Nexc, Nfreq),         complex,   # D1 / inv_D1 (×2)
              (Nexc, Nfreq),         complex,
              (Nmodes, Nexc, Nfreq), complex,   # inv_D2
              (Nmodes, Nexc, Nexc),  complex,   # num_full
              (Nmodes, Nexc, Nfreq), complex,   # temp
              (Nmodes, Nfreq),       complex,   # d2
              (Nmodes, Nfreq),       complex)   # d3
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) RAM — D1, inv_D1, inv_D2, num_full, temp, d2, d3 — total {ram:.3f} GB')

    t0 = time.perf_counter()
    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) - vectorized over modes and excitons')

    # P_outer[s1, s2] = pa[s1] * pb[s2].conj  (Nexc, Nexc)
    P_outer = pos_operator_list[ialpha][:, np.newaxis] * pos_operator_list[ibeta][np.newaxis, :].conjugate()
    P_diag  = np.diag(P_outer)  # (Nexc,)

    # D1[s, f] = Ex[f] - exc_energies[s] + i*gamma         (Nexc, Nfreq)
    D1     = Ex[np.newaxis, :] - exc_energies[:, np.newaxis] + 1j * gamma
    inv_D1 = 1.0 / D1

    # inv_D2[m, s, f] = 1 / (D1[s, f] + freqs_eV[m])      (Nmodes, Nexc, Nfreq)
    inv_D2 = 1.0 / (D1[np.newaxis] + freqs_eV[:, np.newaxis, np.newaxis])

    # exc_ph_diag[m, s] = exc_ph[m, s, s]                  (Nmodes, Nexc)
    exc_ph_diag = np.einsum('mii->mi', exc_ph)

    # d2: sum_s P_diag[s] * exc_ph_diag[m,s] * inv_D1[s,f] * inv_D2[m,s,f]
    num_d2 = P_diag[np.newaxis, :] * exc_ph_diag            # (Nmodes, Nexc)
    d2 = -(num_d2[:, :, np.newaxis] * inv_D1[np.newaxis] * inv_D2).sum(axis=1)  # (Nmodes, Nfreq)

    # d2 + d3: factored matmul avoids (Nmodes, Nexc, Nexc, Nfreq)
    #   step 1: temp[m,s1,f] = sum_s2 P_outer[s1,s2] * exc_ph[m,s1,s2] * inv_D2[m,s2,f]
    #   step 2: d3[m,f]      = -sum_s1 temp[m,s1,f] * inv_D1[s1,f]
    num_full = P_outer[np.newaxis] * exc_ph   # (Nmodes, Nexc, Nexc)
    temp     = np.matmul(num_full, inv_D2)    # (Nmodes, Nexc, Nfreq)
    d3 = -(temp * inv_D1[np.newaxis]).sum(axis=1)  # (Nmodes, Nfreq)

    print(f'  (alpha={ialpha+1}, beta={ibeta+1}) done in {time.perf_counter() - t0:.3f} s')
    return d2, d3

parser = argparse.ArgumentParser()
parser.add_argument('--h5_file', default='exciton_phonon_couplings.h5')
parser.add_argument('--dip_mom_file_b1', default='eigenvalues_b1.dat')
parser.add_argument('--dip_mom_file_b2', default='eigenvalues_b2.dat')
parser.add_argument('--dip_mom_file_b3', default='eigenvalues_b3.dat')
parser.add_argument('--dE', type=float, default=0.002, help='Energy step in eV for the Ex grid')
parser.add_argument('--gamma', type=float, default=0.05, help='Broadening parameter in eV')
parser.add_argument('--vectorized_flavor', type=int, default=2, choices=[0, 1, 2], help='0: no vectorization (triple loop), 1: vectorize over excitons only, 2: vectorize over both excitons and modes (more memory usage but faster)')
parser.add_argument('--test_functions', action='store_true', help='Run all three implementations on Nexc=10 and check they agree')
parser.add_argument('--freqs_file', default='freqs.dat', help='File containing phonon frequencies in cm^-1')
parser.add_argument('--limit_Nexc', type=int, default=None, help='Limit number of excitons to load (for testing)')
args = parser.parse_args()

h5_file = args.h5_file
dip_mom_file_b1 = args.dip_mom_file_b1
dip_mom_file_b2 = args.dip_mom_file_b2
dip_mom_file_b3 = args.dip_mom_file_b3
vectorized_flavor = args.vectorized_flavor
test_functions = args.test_functions
freqs_file = args.freqs_file
dE = args.dE
Emin = 0.0  # Minimum excitation energy in eV
gamma = args.gamma  # eV
limit_Nexc = args.limit_Nexc # int or None

flavor_desc = {0: 'no vectorization (triple loop)',
               1: 'vectorized over excitons',
               2: 'vectorized over excitons and modes'}
print('--- Options ---')
print(f'  h5_file           : {h5_file}')
print(f'  freqs_file        : {freqs_file}')
print(f'  dip_mom_file_b1   : {dip_mom_file_b1}')
print(f'  dip_mom_file_b2   : {dip_mom_file_b2}')
print(f'  dip_mom_file_b3   : {dip_mom_file_b3}')
print(f'  dE                : {dE} eV')
print(f'  gamma             : {gamma} eV')
print(f'  vectorized_flavor : {vectorized_flavor} ({flavor_desc[vectorized_flavor]})')
print(f'  test_functions    : {test_functions}')
print(f'  limit_Nexc        : {args.limit_Nexc} (None means no limit)')
print('---------------\n')

rec_cm_to_eV = 1.239841984e-4  # Conversion factor from cm^-1 to eV
freqs_rec_cm = np.loadtxt(freqs_file)  # Load phonon frequencies in cm^-1
freqs_eV = freqs_rec_cm * rec_cm_to_eV  # Convert frequencies. shape (Nmodes,)


print(f'Reading data from {h5_file}')
with h5py.File(h5_file, 'r') as hf:
    # rpa_diag_data = hf['rpa_diag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
    exc_ph = hf['rpa_offdiag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
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
Nfreq = Ex.shape[0] # number of energies for incident light

if test_functions:
    print('Test mode: truncating to Nexc=10')
    Nexc = 10
    exc_ph = exc_ph[:, :Nexc, :Nexc]
    exc_energies = exc_energies[:Nexc]
    pos_operator_list = [p[:Nexc] for p in pos_operator_list]

    print('Running all three implementations...')
    results = {}
    for flavor, fn in [(0, calculate_tensor_not_vectorized),
                       (1, calculate_tensor_vectorize_over_excitons),
                       (2, calculate_tensor_vectorized_over_modes_and_excitons)]:
        d2_all = np.zeros((3, 3, Nmodes, Nfreq), dtype=complex)
        d3_all = np.zeros((3, 3, Nmodes, Nfreq), dtype=complex)
        for ialpha in range(3):
            for ibeta in range(3):
                d2_all[ialpha, ibeta], d3_all[ialpha, ibeta] = fn(ialpha, ibeta)
        results[flavor] = (d2_all, d3_all)

    ref_d2, ref_d3 = results[0]
    all_passed = True
    for flavor in [1, 2]:
        d2, d3 = results[flavor]
        err_d2 = np.max(np.abs(d2 - ref_d2))
        err_d3 = np.max(np.abs(d3 - ref_d3))
        status_d2 = 'PASS' if err_d2 < 1e-10 else 'FAIL'
        status_d3 = 'PASS' if err_d3 < 1e-10 else 'FAIL'
        print(f'flavor {flavor} vs flavor 0 — d2: {status_d2} (max|diff|={err_d2:.2e}),  d3: {status_d3} (max|diff|={err_d3:.2e})')
        if status_d2 == 'FAIL' or status_d3 == 'FAIL':
            all_passed = False
    if all_passed:
        print('All tests passed.')
    else:
        print('Some tests FAILED.')
    raise SystemExit(0)

if limit_Nexc is not None:
    print(f'Limiting to Nexc={limit_Nexc} excitons as specified by --limit_Nexc')
    exc_ph = exc_ph[:, :limit_Nexc, :limit_Nexc]
    exc_energies = exc_energies[:limit_Nexc]
    pos_operator_list = [p[:limit_Nexc] for p in pos_operator_list]

alpha_tensor_d2 = np.zeros((3, 3, Nmodes, Nfreq), dtype=complex)
alpha_tensor_d3 = np.zeros((3, 3, Nmodes, Nfreq), dtype=complex)

for ialpha in range(3):
    for ibeta in range(3):
        if vectorized_flavor == 2:
            d2, d3 = calculate_tensor_vectorized_over_modes_and_excitons(ialpha, ibeta)
        elif vectorized_flavor == 1:
            d2, d3 = calculate_tensor_vectorize_over_excitons(ialpha, ibeta)
        else:
            d2, d3 = calculate_tensor_not_vectorized(ialpha, ibeta)
        alpha_tensor_d2[ialpha, ibeta] = d2
        alpha_tensor_d3[ialpha, ibeta] = d3

# save results to h5 file
output_h5_file = 'susceptibility_tensors_first_order.h5'
with h5py.File(output_h5_file, 'w') as hf:
    hf.create_dataset('excitation_energies', data=Ex)
    hf.create_dataset('alpha_tensor_d2', data=alpha_tensor_d2) # shape (3, 3, Nmodes, Ndata)
    hf.create_dataset('alpha_tensor_d3', data=alpha_tensor_d3) # shape (3, 3, Nmodes, Ndata)
