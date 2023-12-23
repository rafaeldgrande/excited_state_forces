
import numpy as np

eigvecs_file = 'eigvecs' # eigvecs file with phonon frequencies and dynamical matrix eigenvectors. This file is produced by dynmat.x through the variable fileig
atomic_pos_file = 'Atoms_info' # file with atomic species and their masses in a.u.. The format of the file is the following:
#  Li   0.000000000   0.000000000   0.000000000
#  F    -2.02948455    2.02948455    2.02948455

T = 300 # temperature in K
seed = 1234

# Tolerance for values to be considered zero
zero_tol = 1e-6

# Physical constants and conversion factors
c = 2.99792458e10 # cm/s speed of light
Na = 6.02214076e23 # Avogadro's number

ry2ev = 13.605703976
ev2ry = 1/ry2ev
bohr2ang = 0.529177249
ang2bohr = 1/bohr2ang
rec_cm2eV = 1.23984198e-4
eV2J = 1.602176634e-19
J2eV = 1/eV2J
m2ang = 1e10
kb = 8.61733326e-5 # boltzmann constant in eV/K

def read_eigvecs_file(eigvecs_file, Natoms):

    print('Reading eigvecs file')
    arq_eigvecs = open(eigvecs_file)

    eigvecs = np.zeros((3*Natoms, 3*Natoms))
    i_eigvec = -1
    freqs = [] # frequencies in cm-1
    for line in arq_eigvecs:
       line_split = line.split()
       if len(line_split) > 0:
          if line_split[0] == 'freq':
             freqs.append(float(line_split[-2]))
             i_eigvec += 1
             i_atom_dir = -1
          if line_split[0] == '(':
             for i_dir in [1,3,5]:
                i_atom_dir += 1
                # eigvecs[-1].append(float(line_split[i_dir]))
                eigvecs[i_atom_dir, i_eigvec] = float(line_split[i_dir])

    # for i_eigvec in range(len(eigvecs)):
    #    eigvecs[i_eigvec] = np.array(eigvecs[i_eigvec])
    #    # print('Norm = ', np.dot(eigvecs[i_eigvec], eigvecs[i_eigvec]))
    
    arq_eigvecs.close()
    print('Finished reading eigvecs file. Obtained frequencies and eigenvectors')
    
    return np.array(freqs), eigvecs


# read atomic positions
#  Li   0.000000000   0.000000000   0.000000000
#  F    -2.02948455    2.02948455    2.02948455

atomic_pos, atomic_simb = [], []
Masses = []

arq = open(atomic_pos_file)
for line in arq:
    line_split = line.split()
    x, y, z = float(line_split[2]), float(line_split[3]), float(line_split[4])
    atomic_pos.append(x)
    atomic_pos.append(y)
    atomic_pos.append(z)
    atomic_simb.append(line_split[0])
    Masses.append(float(line_split[1]))
arq.close()

Natoms = len(atomic_simb) 
atomic_pos = np.array(atomic_pos)

# load eigvecs from eigvecs file
freqs, eigvecs = read_eigvecs_file(eigvecs_file, Natoms)

# convert phonon frequencies to rad/s
freq_rad_per_s = 2*np.pi*c*freqs

# we approximate Dij = Kij / sqrt(m_i * m_j) (just first neighbours interactions included)
# use spectral decomposition to create the dynamical matrix
# eigenvalues of D are omega**2
print(f"Creating dynamical matrix from eigenvectors using spectral decomposition.")
dyn_mat_from_eigvecs = eigvecs @ np.diag(np.sign(freq_rad_per_s) * freq_rad_per_s**2) @ eigvecs.T
# here multiplicating by np.sign(freq_rad_per_s) to preserve the sign of negative frequencies
    
# reinforce the matrix to be symmetric
print(f"Reinforcing the matrix to be symmetric. D = (D+D.T)/2")
dyn_mat_from_eigvecs = (dyn_mat_from_eigvecs + dyn_mat_from_eigvecs.T)/2


# create force constant matrix
# Kij = Dij * sqrt(m_i * m_j)
# Dij is in units of (rad/s)^2. 
# I want Kij to be in units of eV/angs^2
# Let's work with m in a.u. -> 1 a.u. = 1e-3 kg/Na = 1-3/(6.022*10^23) kg
# So the unit of K is kg * rad^2 / s^2 = J / m^2
print(f"Creating force constant matrix from the dynamical matrix.")
force_constant_mat = np.zeros((3*Natoms, 3*Natoms))
for i_atom in range(Natoms):
    for i_dir in range(3):
        i_ind = 3*i_atom + i_dir
        for j_atom in range(Natoms):
            for j_dir in range(3):
                j_ind = 3*j_atom + j_dir
                force_constant_mat[i_ind, j_ind] = dyn_mat_from_eigvecs[i_ind, j_ind] * np.sqrt(Masses[i_atom] * Masses[j_atom]) 

# converting J/m^2 to eV/angs^2
force_constant_mat = force_constant_mat * J2eV / m2ang**2

# compute eigenvalues and eigenvectors of force constant matrix
# cannot invert force constant matrix as some of its
# eigenmodes have zero value
print(f"Calculating the eigenvectors |ui> and eigenvalues lambda_i of the force constant matrix.")
eigenvalues, eigenvectors = np.linalg.eigh(force_constant_mat)

# How many acoustic modes?
print(f"There are {np.count_nonzero(abs(eigenvalues) < zero_tol)} acoustic modes")

# create displacements vector
displacements = np.zeros(3*Natoms)

# apply random displacements delta_x = np.random.normal(mu = 0, sigma) * u_i
# where sigma = sqrt(kb * T / lambda_i) and lambda_i is an eigenvalue of the force constant matrix
# and u_i is an eigenvector of the force constant matrix 
for i_eigvec in range(len(eigenvalues)):
    if eigenvalues[i_eigvec] > zero_tol:
        print(f'i_eigvec = {i_eigvec} and <x^2> = {np.sqrt(kb * T / eigenvalues[i_eigvec]):.8f}')
        delta_x = np.random.normal(0, np.sqrt(kb * T / eigenvalues[i_eigvec]))
        displacements += eigenvectors[i_eigvec] * delta_x   

print(f'Modulus of 3N displacements = {np.linalg.norm(displacements):.8f} angstroms')

# final pos
final_pos = displacements + atomic_pos

# write final pos
arq = open('atomic_disp_rand_displacements', 'w')

for i_atom in range(Natoms):
    x = final_pos[3*i_atom]
    y = final_pos[3*i_atom + 1 ]
    z = final_pos[3*i_atom + 2]
    arq.write(f"{atomic_simb[i_atom]} {x:.8f} {y:.8f} {z:.8f}\n")

arq.close()

print('Finished.')
