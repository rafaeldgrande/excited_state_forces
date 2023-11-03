

# standard values   
# TODO: make the code read a configuration file
file_out_QE = 'out'

excited_state_forces_file = 'forces_cart.out-1'
flavor = 2

reinforce_CM_forces_null = True
CM_disp_null = True

eigvecs_file = 'eigvecs'
atoms_file = 'atoms'

# Initial message
print('####################################')
print('Starting harmonic extrapolation code')
print('####################################\n\n')

print('Parameters:')
print(f'file_out_QE: {file_out_QE}')
print('   output file from QE (pw.x, bands.x) that contains the dft forces on atoms. If not provided, DFT forces are assumed to be null')
print(f'excited_state_forces_file: {excited_state_forces_file}')
print(f'   output file from excited state forces code')
print(f'flavor: {flavor}')
print(f'   which flavor of excited state forces are we using? 1 - RPA_diag (IBL without kernel) or 2 - RPA_diag_offdiag (from prof. Strubbe thesis)')
print(f'CM_disp_null: ', CM_disp_null)
print('    If true the displacements are such that the system center of mass does not move. (In future, I will fix the center of mass of a set of atoms, for example fix the center of mass of MA cation on MAPbI3!)')
print('End of parameters \n')
# TODO In future, I will fix the center of mass of a set of atoms, for example fix the center of mass of MA cation on MAPbI3!

import numpy as np
import subprocess

''' 

This code computes the displacement due to the excited 
state forces + dft forces, using the harmonic approximation 
that the force constant matrix does not vary substantially 
during this displacement.

The force constant matrix is defined as

K_{ia,jb} \approx -\delta F_{ia} / \deltat x_{jb}

where i,j represents the atoms indexes and a,b the 
cartesian directions x, y and z. In matrix notation this 
equation can be written as

F = K x 

Here F = F_dft + F_excited

We cannot invert this matrix as some of its eigenvalues are null (acoustic modes)
To solve this we use spectral decomposition -> k = sum_i |u_i> lambda_i <u_i|
where lambda_i and |u_i> are eigenvalues and eigenvectors of k.

Expressing the displacements and force in terms of the eigenvectors

|x> = sum_i |u_i> x_i
|F> = sum_i |u_i> F_i

If the force on the center of mass is null, then F_i = 0 when lambda_i = 0

Then x_i is given by

x_i = F_i / lambda_i if lambda_i != 0
x_i = 0 if lambda_i = 0

'''

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


def read_dft_forces_qe(file, Natoms):
    
    # dft forces array - Ry/bohr
    dft_forces = np.zeros((3*Natoms))
    
    # getting data from QE output file
    grep_command = ["grep", "force", file]
    
    # result = subprocess.run(grep_command, stdout=subprocess.PIPE)
    # print(result.stdout)
    try:
        grep_output = subprocess.check_output(grep_command, stderr=subprocess.PIPE, text=True)
        print('Grep worked sucessfully!')
    except subprocess.CalledProcessError as e:
        print("Error executing grep:", e)
        print("Did not find the DFT forces!")
        print("DFT forces are set to 0.\n")
        return dft_forces
        
    # filtering - gtting the first Natoms lines
    temp_text = grep_output.split('\n')[:Natoms]
    
    # parsing data
    for iatom in range(Natoms):
        # '     atom    1 type  1   force =     0.00000000    0.00000000    0.02862271'
        line_split = temp_text[iatom].split()
        fx, fy, fz = float(line_split[-3]), float(line_split[-2]), float(line_split[-1])
        
        dft_forces[iatom * 3] = fx
        dft_forces[iatom * 3 + 1] = fy
        dft_forces[iatom * 3 + 2] = fz

    print("")
        
    return dft_forces

def read_excited_forces(excited_state_forces_file, flavor):
    # flavor = 1 -> RPA_diag 
    # flavor = 2 -> RPA_diag_offiag 
    # flavor = 3 -> RPA_diag_Kernel
    
    data = np.loadtxt(excited_state_forces_file, usecols=flavor+1)
    return data
    
def sum_comp_vec(vector, dir):
    index_start = ['x', 'y', 'z'].index(dir)
    sum_dir = np.sum(vector[index_start:3*Natoms:3])
    return sum_dir

def CM_force(force_vector):
    
    print('Checking if force on center of mass is zero. \n')

    CM_force_null = True
    
    for dir in ['x', 'y', 'z']:

        sum_dir = abs(sum_comp_vec(force_vector, dir))

        print(f'Sum of {dir} components = {sum_dir:.8f} eV/angstrom')
        if sum_dir >= zero_tol:
            print('WARNING: Does not obey ASR!')
            CM_force_null = False

    if CM_force_null == False:
        print('Force on center of mass is not zero. \n')
        if reinforce_CM_forces_null == True:
            print('Reinforcing that force on center of mass is null. \n')
            return ASR_on_vector(force_vector)
        else:
            print('Not enforcing that force on center of mass is null. Check reinforce_CM_forces_null variable\n')

def ASR_on_vector(vector):
    
    new_vector = np.zeros(vector.shape)
    
    for dir in ['x', 'y', 'z']:
        index_start = ['x', 'y', 'z'].index(dir)

        sum_dir = sum_comp_vec(vector, dir)
        
        for iatom in range(Natoms):
            new_vector[index_start+iatom*3] = vector[index_start+iatom*3] - sum_dir / Natoms
            
    return new_vector


def estimate_energy_change(displacements, excited_forces, force_constant_mat):
    
    ''' The DFT energy approximatelly is given by
    
    Edft(x) = E0 + 1/2 k (x-x0)**2
    where x0 is the DFT equilibrium position vector or expanded around xi
    Edft(x) = Ei + k(xi-x0)(x-xi) + 1/2 k (x-xi)**2
    xi is the initial position vector, Ei = E0 + 1/2 k (x-x0)**2
    and -k(xi-x0) is equal to the DFT force when x=xi, so
    Edft(x) = Ei + Fdft_i(x-xi) + 1/2 k (x-xi)**2
    
    The excited state energy is given by
    Omega(x) = Omega_i - Fexc_i (x-xi)

     
    '''

    # Estimates the DFT energy difference from x_eq to x_i
    Delta_E_dft_i_to_eq = np.real(displacements.T @ force_constant_mat @ displacements / 2)
    
    # then estimates the Exciton energy difference from x_eq to x_i
    # in this case the force is approximatelly constant so Delta E = - displacement . force
    Delta_Omega_i_to_eq = np.real(- np.dot(displacements, excited_forces))
      
    print(f"""Energy changes
x_i = initial position vector
x_eq = Equilibrim position for Edft + Omega in the harmonic approximation
          
DFT energy change due to displacements: E(x_eq) - E(x_i) = {Delta_E_dft_i_to_eq:.8f} eV
Excitation energy change due to displacements: Omega(x_eq) - Omega(x_i) = {Delta_Omega_i_to_eq:.8f} eV
Total energy change: E+Omega(x_eq) - E+Omega(x_i) = {(Delta_E_dft_i_to_eq + Delta_Omega_i_to_eq):.8f} eV""")
    
def are_parallel(vector1, vector2, tolerance=1e-6):
    """
    Check if two N-dimensional arrays (vectors) are parallel.

    Parameters:
    - vector1: NumPy array, the first vector.
    - vector2: NumPy array, the second vector.
    - tolerance: Float, a small positive value to account for numerical imprecisions.

    Returns:
    - True if the vectors are parallel, False otherwise.
    """

    # Check if the vectors have the same shape
    if vector1.shape != vector2.shape:
        return False

    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes (lengths) of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Check if the dot product is within a small tolerance of the expected value
    expected_dot_product = magnitude1 * magnitude2
    ratio = dot_product / expected_dot_product
    if abs(dot_product - expected_dot_product) < tolerance:
        return f'Yes!.  <u.v>/(|v||u|) = {ratio:.6f}'
    else:
        return f'No!.  <u.v>/(|v||u|) = {ratio:.6f}'
    
def print_info_displacements(displacements):
    
    print('Printing displacement and its modulus for each atom! Units: angstroms')
    print('iatom  rx  ry  rz  |r|')
    
    rmod = []

    for iatom in range(Natoms):
        r = displacements[3*iatom:3*(iatom+1)]
        print(f' {iatom+1}  {r[0]:.6f}   {r[1]:.6f}   {r[2]:.6f}   {np.linalg.norm(r):.6f}')
        rmod.append(np.linalg.norm(r))
        
    print(f"Max displacement {max(rmod):.6f} angstroms for atom {rmod.index(max(rmod))+1}")
    print(f"Min displacement {min(rmod):.6f} angstroms for atom {rmod.index(min(rmod))+1}")
    print(f"Mean atomic displacements: {np.mean(rmod):.6f} angstroms")
    print(f"Modulus of 3N displacement vector: {np.linalg.norm(displacements):.8f} angstroms")
        
    print("")
    
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

def write_displacements(displacements, arq_name):
    print(f"Writing {arq_name} file")
    arq_out = open(arq_name, 'w')

    for iatom in range(Natoms):
        r = displacements[3*iatom:3*(iatom+1)]
        arq_out.write(f"{iatom+1}    {r[0]:.8f}     {r[1]:.8f}    {r[2]:.8f} \n")

    arq_out.close()
    
def load_atoms(atoms_file):
    # Mo 95.95
    # S  32.06
    # S  32.06
    
    Atoms, Masses = [], []
    arq = open(atoms_file)
    for line in arq:
        linha = line.split()
        Atoms.append(linha[0])
        # masses in kg -> 1a.u. corresposnts to 1g/mol
        Masses.append(float(linha[1]) * 1e-3 / Na)
        
    return Atoms, Masses
        

# Loading atoms informations
Atoms, Masses = load_atoms(atoms_file)
Natoms = len(Atoms)

# loading dft forces
dft_forces = read_dft_forces_qe(file_out_QE, Natoms)     

# converting from ry/bohr to eV/angs
dft_forces = dft_forces * ry2ev / bohr2ang

# loading excited state forces - already in eV/angs
excited_forces = read_excited_forces(excited_state_forces_file, flavor)

# checking if excited state forces obey ASR
# check_ASR_vector(excited_forces)

# # reinforce ASR on excited state force vector
# if reinforce_ASR_excited_state_forces == True:
#     excited_forces = ASR_on_vector(excited_forces)
    
f_tot = dft_forces + excited_forces
    
# load eigvecs from eigvecs file
freqs, eigvecs = read_eigvecs_file(eigvecs_file, Natoms)
    
## normalize eigenvectors
#for i_eigvec in range(len(eigvecs)):
#    eigvecs[i_eigvec] = eigvecs[i_eigvec] / np.linalg.norm(eigvecs[i_eigvec])
    
# convert phonon frequencies to rad/s
freq_rad_per_s = 2*np.pi*c*freqs
    
# we approximate Dij = Kij / sqrt(m_i * m_j) (just first neighbours interactions included)
# use spectral decomposition to create the dynamical matrix
# eigenvalues of D are omega**2
print(f"Creating dynamical matrix from eigenvectors using spectral decomposition.")
dyn_mat_from_eigvecs = eigvecs @ np.diag(freq_rad_per_s**2) @ eigvecs.T
    
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

# reinforce matrix to be symmetric
# force_constant_mat = (force_constant_mat + force_constant_mat.T)/2
    
# cannot invert force constant matrix as some of its
# eigenmodes have zero value. we solve F=kx, by
# using k eigenvectors and eigenvalues
print(f"Calculating the eigenvectors |ui> and eigenvalues lambda_i of the force constant matrix.")
eigenvalues, eigenvectors = np.linalg.eigh(force_constant_mat)

# now we obtain the forces in eigenvecs basis
print(f"Projecting forces in eigenvecs basis.")
forces_eigvecs_basis = np.zeros((f_tot.shape))

for i_eigvec in range(len(eigenvectors)):
    forces_eigvecs_basis[i_eigvec] = np.dot(eigenvectors[:, i_eigvec], f_tot)

# now we calculate the displacement in eigvecs basis
# if the phonon frequency is null (acoustic mode) then 
# this component is null

print("Calculating displacements in eigvecs basis. <ui|x> = <ui|F> / lambda_i for lambda_i != 0. <ui|x> = 0 otherwise.")
disp_eigvecs_basis = np.zeros((f_tot.shape))

for i_eigvec in range(len(eigenvectors)):
    if abs(eigenvalues[i_eigvec]) > zero_tol:
        disp_eigvecs_basis[i_eigvec] = forces_eigvecs_basis[i_eigvec] / eigenvalues[i_eigvec]

# now we calculate the displacements in cartesian basis
print("Projecting displacements in cartesian basis.")
disp_cart_basis = np.zeros((f_tot.shape))

for i_eigvec in range(len(eigenvectors)):
    if abs(eigenvalues[i_eigvec]) > zero_tol:
        disp_cart_basis += disp_eigvecs_basis[i_eigvec] * eigenvectors[:, i_eigvec] 


    
write_displacements(disp_cart_basis, 'displacements_cart.dat')
# write_displacements(displacements_dft, 'displacements_Fdft.dat')


print("\n\nSome extra information:")
# printing information about DFT and Excited state forces

print(f"\nModulus of 3N DFT forces vector: {np.linalg.norm(dft_forces):.8f} eV/angstrom")
print(f"Modulus of 3N Excited state forces vector: {np.linalg.norm(excited_forces):.8f} eV/angstrom")
print(f"Modulus of 3N DFT + Excited state (total force) force vector: {np.linalg.norm(f_tot):.8f} eV/angstrom\n")

print('Is (Fdft + Fexcited) parallel to displacement? ', are_parallel(disp_cart_basis, f_tot, tolerance=1e-2))
print('Is F_excited parallel to displacement? ', are_parallel(disp_cart_basis, excited_forces, tolerance=1e-2))
print("")
print_info_displacements(disp_cart_basis)
estimate_energy_change(disp_cart_basis, excited_forces, force_constant_mat)
print("")


print('####################################')
print('Finished!')
print('####################################\n\n')
