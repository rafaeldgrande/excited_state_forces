
import numpy as np
import sys

""" Apply first order perturbation theory on eigenvalues and dipole moments and 
    and produces the file eigenvalues_perturbed.dat
    
    It reads the matrix elements indexes from file exciton_pairs.dat and then looks 
    for files forces_i_j.dat. If file is not found, then it assumes that <i|dH|j> = 0
    
    It also reads the file displacements.dat and then calculate the direct product
    <i|dH \cdot dr |j>. 
    
    Inputs: eigenvalues.dat, exciton_pairs.dat, displacements.dat
    Output: eigenvalues_perturbed.dat
    
    Usage: ./first_order_pert_on_eigvals_dip_moments.py eigenvalues.dat exciton_pairs.dat displacements.dat exciton_concentration flavor_forces eigenvalues_perturbed.dat 
    """
    
TOL_DEG = 1e-6
limit_eigenvalues = 2000 

# Files to be read 
eigenvalues_file = sys.argv[1]
exciton_pairs_file = sys.argv[2]
displacements_file = sys.argv[3]
exciton_concentration = float(sys.argv[4])
flavor_forces = int(sys.argv[5])

# Output file
output_file = sys.argv[6]


# Read the eigenvalues and dipole moments from the file
data = np.loadtxt(eigenvalues_file)
eigvals0 = data[:limit_eigenvalues,0] 
dip_moments0 = data[:limit_eigenvalues,2] + 1j*data[:limit_eigenvalues,3]

eigvals_pert = np.zeros(eigvals0.shape)
dip_moments_pert = np.zeros(dip_moments0.shape, dtype=complex)

displacement = np.loadtxt(displacements_file)
displacement = displacement

# Read the exciton pairs
exciton_pairs = []
with open(exciton_pairs_file) as f:
    for line in f:
        line_split = line.split()
        if len(line_split) == 1:
            i, j = int(line_split[0]), int(line_split[0])
        else:
            i, j = int(line_split[0]), int(line_split[1])
        exciton_pairs.append((i, j))



# load forces

exciton_phonon_matrix = np.zeros((limit_eigenvalues, limit_eigenvalues), dtype=complex)
 
for exciton_pair in exciton_pairs:
    i, j = exciton_pair
    forces_file = f"forces_cart.out_{i}_{j}"
    force = np.loadtxt(forces_file, usecols=(1+flavor_forces), dtype=complex)
    coupling = -np.dot(force, displacement) * exciton_concentration
        
    exciton_phonon_matrix[i-1, j-1] = coupling
    if i != j:
        exciton_phonon_matrix[j-1, i-1] = np.conj(coupling)

exciton_phonon_diags = np.diag(exciton_phonon_matrix).real
print(exciton_phonon_diags[:10])

# Apply correction on energy
eigvals_pert = eigvals0 + exciton_phonon_diags

# Apply correction on dipole moments
for iexc in range(limit_eigenvalues):
    t1 = 0.0
    t2 = 0.0
    for jexc in range(limit_eigenvalues):
        if abs(eigvals0[iexc] - eigvals0[jexc]) > TOL_DEG:
            t1 += exciton_phonon_matrix[jexc, iexc] * dip_moments0[jexc] / (eigvals0[iexc] - eigvals0[jexc])
            t2 += dip_moments0[jexc] * np.abs(exciton_phonon_matrix[jexc, iexc])**2 / (eigvals0[iexc] - eigvals0[jexc])**2
    # print('iexc = ',iexc)
    # print('t1 = ', t1)
    # print('t2 = ', t2)
    dip_moments_pert[iexc] = (dip_moments0[iexc] + t1) / (1 + t2)

print(np.abs(dip_moments0[:10])**2 - np.abs(dip_moments_pert[:10])**2)
    
# Save the perturbed eigenvalues and dipole moments
np.savetxt(output_file, np.column_stack((eigvals_pert, np.abs(dip_moments_pert)**2, dip_moments_pert.real, dip_moments_pert.imag)), fmt="%.9f")

print('Finished')
