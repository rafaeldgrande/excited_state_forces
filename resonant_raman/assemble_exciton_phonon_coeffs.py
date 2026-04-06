
import numpy as np
import h5py

exciton_pairs_file = 'exciton_pairs.dat'
pair_indexes_to_load = []

def read_excited_state_forces_file_ph(filename):
    # read excited state forces in ph basis
    data = np.loadtxt(filename, dtype=complex)
    rpa_diag = data[:, 1]
    rpa_offdiag = data[:, 2]
    rpa_diag_plus_kernel = data[:, 3]
    
    return rpa_diag, rpa_offdiag, rpa_diag_plus_kernel


print('Reading exciton pairs to load from file:', exciton_pairs_file)
arq = open(exciton_pairs_file)
for line in arq:
    pair_indexes_to_load.append(tuple(map(int, line.split())))
arq.close()
print('Total of exciton pairs to be loaded:', len(pair_indexes_to_load))

# maximum indexes to load
max_index = max(max([pair[0] for pair in pair_indexes_to_load]), max([pair[1] for pair in pair_indexes_to_load]))

pair = pair_indexes_to_load[0]
i_exciton_1 = pair[0]
i_exciton_2 = pair[1]
filename = f'forces_ph.out_{i_exciton_1}_{i_exciton_2}'
rpa_diag, rpa_offdiag, rpa_diag_plus_kernel = read_excited_state_forces_file_ph(filename)
Nmodes = rpa_diag.shape[0]
print(f'Number of phonon modes detected: {Nmodes}')

data = np.zeros((3, Nmodes, max_index, max_index), dtype=complex)

print('Loading exciton-phonon coupling data from files forces_ph.out_i_j')
# loading all data

counter = 0
for pair in pair_indexes_to_load:
    i_exciton_1 = pair[0] # starts at 1
    i_exciton_2 = pair[1]
    
    filename = f'forces_ph.out_{i_exciton_1}_{i_exciton_2}'
    rpa_diag, rpa_offdiag, rpa_diag_plus_kernel = read_excited_state_forces_file_ph(filename)
    
    data[0, :, i_exciton_1-1, i_exciton_2-1] = rpa_diag
    data[1, :, i_exciton_1-1, i_exciton_2-1] = rpa_offdiag
    data[2, :, i_exciton_1-1, i_exciton_2-1] = rpa_diag_plus_kernel
    
    # assuming there is only one i, j and not j, i
    if i_exciton_1 != i_exciton_2:
        data[0, :, i_exciton_2-1, i_exciton_1-1] = rpa_diag.conjugate()
        data[1, :, i_exciton_2-1, i_exciton_1-1] = rpa_offdiag.conjugate()
        data[2, :, i_exciton_2-1, i_exciton_1-1] = rpa_diag_plus_kernel.conjugate()
    counter += 1
    
    if counter % 100 == 0:
        print(f'  Loaded {counter} pairs of {len(pair_indexes_to_load)} ({100.0*counter/len(pair_indexes_to_load):.2f} %)')

# saving data in hdf5 file
with h5py.File('exciton_phonon_couplings.h5', 'w') as hf:
    hf.create_dataset('rpa_diag', data=data[0])
    hf.create_dataset('rpa_offdiag', data=data[1])
    hf.create_dataset('rpa_diag_plus_kernel', data=data[2])
    
print('Exciton-phonon coupling data saved in exciton_phonon_couplings.h5')
print('Finished!')