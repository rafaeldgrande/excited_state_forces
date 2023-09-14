

# standard values   
dyn_file = 'dyn'
file_out_QE = 'out'
excited_state_forces_file = 'forces_cart.out-1'
flavor = 2
reinforce_ASR_excited_state_forces = True
reinforce_ASR_dyn_mat = True
CM_disp_null = True

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

F = - K x -> x = - K^{-1} F

Here F = F_dft + F_excited

'''

ry2ev = 13.6057039763
ev2ry = 1/ry2ev
bohr2ang = 0.529177
ang2bohr = 1/bohr2ang
zero_tol = 1e-6

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
        print("DFT forces are set to 0s. This is true if you are starting configurations is the DFT equilibrium")
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
        
    return dft_forces

def check_string_atomic_pos_dyn_file(input_string):
    # Split the string into individual elements
    elements = input_string.split()

    # Check if the string contains at least 5 elements
    if len(elements) < 5:
        return False

    try:
        # Check if the first two elements are integers
        int(elements[0])
        int(elements[1])

        # Check if the last three elements are floats
        float(elements[-3])
        float(elements[-2])
        float(elements[-1])

        return True

    except ValueError:
        return False

def read_dyn_matrix(dyn_file):
    
    # list with masses - used to make the displacements vector
    # to not move the center of mass
    masses = []
    atoms_species = []
    
    # we are reading the dynamical matrix 
    arq = open(dyn_file, "r")
    
    # read the first two lines
    arq.readline()
    arq.readline()
    
    # third line has the info we want!
    # 5   48   0  15.4935509   0.0000000   0.0000000   0.0000000   0.0000000   0.0000000
    line_split = arq.readline().split()
    
    # now I know the number of atoms 
    Natoms = int(line_split[1])
    
    # creating dyn matrix
    dyn_mat = np.zeros((3*Natoms, 3*Natoms), dtype=complex)
    
    # reading the file
    while True:
        
        line = arq.readline()
        line_split = line.split()
        
        # getting masses
        # identifying a line like this
        # line = "         1  'Li  '    6326.33449141718"
        # line.split() = ['1', "'Li", "'", '6326.33449141718']
        # line.split()[1] = "'Li"
        # line.split()[1].split("'") = ['', 'Li']
        # len(line.split()[1].split("'")) = 2
        if len(line_split) >= 2:
            if len(line_split[1].split("'")) == 2:
                masses.append(float(line_split[3]))
            
        # getting atomic species
        if check_string_atomic_pos_dyn_file(line) == True:
            atoms_species.append(int(line_split[1])-1)
        
        
        # now we look for the following line
        # q = (    0.000000000   0.000000000   0.000000000 )
        
        if len(line_split) > 0:
            if line_split[0] == "q":
                
                # ok, we found it. now we read the next lines!
                
                # empty line
                arq.readline()
                
                # reading dyn values
                for iatom_line in range(Natoms**2):
                    line_split = arq.readline().split()
                    i, j = int(line_split[0])-1, int(line_split[1])-1
                    for i_dir in range(3):
                        line_split = arq.readline().split()
                        for j_dir in range(3):
                            dyn_mat[3*i + i_dir, 3*j + j_dir] = float(line_split[j_dir*2]) + float(line_split[j_dir*2 + 1])
                        
                # Now we finished reading it! We can break the while loop
                break    
            
    arq.close()            
            
    return Natoms, dyn_mat, masses, atoms_species

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

def check_ASR_vector(vector):
    
    for dir in ['x', 'y', 'z']:

        sum_dir = abs(sum_comp_vec(vector, dir))

        print(f'Sum of {dir} components = {sum_dir:.6f}')
        if sum_dir >= zero_tol:
            print('   Does not obey ASR! Use reinforce_ASR_excited_state_forces = True')

def ASR_on_vector(vector):
    
    new_vector = np.zeros(vector.shape)
    
    for dir in ['x', 'y', 'z']:
        index_start = ['x', 'y', 'z'].index(dir)

        sum_dir = sum_comp_vec(vector, dir)
        
        for iatom in range(Natoms):
            new_vector[index_start+iatom*3] = vector[index_start+iatom*3] - sum_dir / Natoms
            
    return new_vector

def ASR_dyn_mat(dyn_mat):
    
    print('Reinforcing that dyn_mat is symmetric and obeys ASR!')
    
    new_dyn_mat = np.zeros(dyn_mat.shape, dtype=complex)
    
    # first we garantee that dyn_mat[i,j] == dyn_mat[j,i]
    
    size_mat, _ = dyn_mat.shape
    
    for i in range(size_mat):
        # diagonal matrix elements 
        new_dyn_mat[i, i] = dyn_mat[i, i]
        
        # off diagonal matrix elements
        for j in range(i+1, size_mat):
            if abs(dyn_mat[i, j] - dyn_mat[j, i]) >= zero_tol:
                print(f'Matrix element {i+1}, {j+1} is not equal to {j+1},{i+1}!')
                print(f"{i+1}, {j+1}, {dyn_mat[i, j]}")
                print(f"{j+1}, {i+1}, {dyn_mat[j, i]}")
            new_dyn_mat[i, j] = 0.5 * (dyn_mat[i, j] + dyn_mat[j, i])
            new_dyn_mat[j, i] = new_dyn_mat[i, j]
            
    # now we have to make the sum of all elements in the same collumn (row)
    # to be zero. As the matrix is already symmetric we will work on the rows
    # 
    
    for i in range(size_mat):
        
        for dir in ['x', 'y', 'z']:
            index_start = ['x', 'y', 'z'].index(dir)
            
            # sum of all values of this row - in the direction dir!
            sum_dir = sum_comp_vec(new_dyn_mat[i,:], dir)
            
            # how many non values there are in this row? in the direction dir!
            non_zero_vals = np.sum(abs(new_dyn_mat[i,index_start:3*Natoms:3]) > zero_tol)
            
            # correct the values in the direction dir
            if abs(sum_dir) >= zero_tol:
                for j in np.arange(index_start, 3*Natoms, 3):
                    # just correct non-zero values!
                    if abs(new_dyn_mat[i,j]) >= zero_tol:
                        new_dyn_mat[i,j] = new_dyn_mat[i,j] - sum_dir / non_zero_vals
                
    return new_dyn_mat

def alternative_inversion(M):
    
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Check if eigenvalues are close to zero (numerically)
    close_to_zero = np.isclose(eigenvalues, 0.0, atol=1e-10)

    if np.any(close_to_zero):
        print("Matrix is not invertible due to near-zero eigenvalues.")
    else:
        # Calculate the inverse using eigenvalue decomposition
        inverse_M = eigenvectors @ np.diag(1.0 / eigenvalues) @ eigenvectors.T

        # print("Inverse of M:")
        # print(inverse_M)
        
    return inverse_M

def make_CM_disp_null(displacements, masses, atoms_species):
    
    # first we calculate the total mass in this unit cell
    Mtot = 0
    for i_species in atoms_species:
        Mtot += masses[i_species]
        
    # calculating the center of mass displacement
    D = np.zeros((3))
    for iatom in range(Natoms):
        r = displacements[3*iatom:3*(iatom+1)]
        mass = masses[atoms_species[iatom]]
        D += r * mass 
    
    D = D / Mtot
    
    # now creating the Dreplicated, which is the same displacement
    # for all atoms. Ex: D = [1,2,3], Drecplicated = [1,2,3,1,2,3,1,2,3...]
    Dreplicated = np.tile(D, Natoms)
    
    return displacements - Dreplicated

def estimate_energy_change(displacements, displacements_dft, excited_forces, dyn_mat):
    
    ''' The DFT energy approximatelly is given by
    
    Edft(x) = E0 + 1/2 k (x-x0)**2
    where x0 is the DFT equilibrium position vector or expanded around xi
    Edft(x) = Ei + k(xi-x0)(x-xi) + 1/2 k (x-xi)**2
    xi is the initial position vector, Ei = E0 + 1/2 k (x-x0)**2
    and -k(xi-x0) is equal to the DFT force when x=xi, so
    Edft(x) = Ei + Fdft_i(x-xi) + 1/2 k (x-xi)**2
    
    The excited state energy is given by
    Omega(x) = Omega_i - Fexc_i (x-xi)
    
    we calculated here x_eq that is due 
    '''
    
    # first estimate the difference of DFT energy = x.T K x 
    Delta_E_dft_i_to_0 = np.real(displacements_dft.T @ dyn_mat @ displacements_dft / 2)
    
    # then estimates the DFT energy difference from x_eq to x_i
    Delta_E_dft_i_to_eq = np.real(displacements.T @ dyn_mat @ displacements / 2)
    
    # then estimates the Exciton energy difference from x_eq to x_i
    # in this case the force is approximatelly constant so Delta E = - displacement . force
    Delta_Omega_i_to_eq = np.real(- np.dot(displacements, excited_forces))
      
    print("""
          
x_i = initial position vector
x_0 = Equilibrium position for DFT surface energy
x_eq = Equilibrim position for Edft + Omega                                                                                                                                                               
          
          """)
    print(f"How far from DFT minimum we are: E(xi) - E(x0) = {Delta_E_dft_i_to_0:.8f} eV.")
    print(f"DFT energy change due to displacements: E(x_eq) - E(xi) = {Delta_E_dft_i_to_eq:.8f} eV")
    print(f"Excitation energy change due to displacements: Omega(x_eq) - Omega(xi) = {Delta_Omega_i_to_eq:.8f} eV")
    print(f"Total energy change: E+Omega(x_eq) - E+Omega(xi) = {(Delta_E_dft_i_to_eq + Delta_Omega_i_to_eq):.8f} eV")
    
                                                                                      
# loading dynamical matrix
Natoms, dyn_mat, masses, atoms_species = read_dyn_matrix(dyn_file)

# converting from ry/bohr^2 to eV/angs^2
dyn_mat = dyn_mat * ry2ev / bohr2ang**2

# apply acoustic sum rule on dyn_mat
if reinforce_ASR_dyn_mat == True:
    dyn_mat = ASR_dyn_mat(dyn_mat)

# loading dft forces
dft_forces = read_dft_forces_qe(file_out_QE, Natoms)     

# converting from ry/bohr to eV/angs
dft_forces = dft_forces * ry2ev / bohr2ang

# loading excited state forces - already in eV/angs
excited_forces = read_excited_forces(excited_state_forces_file, flavor)

# checking if excited state forces obey ASR
check_ASR_vector(excited_forces)

# reinforce ASR on excited state force vector
if reinforce_ASR_excited_state_forces == True:
    excited_forces = ASR_on_vector(excited_forces)

# Now calculating displacements
# x = K^{-1} F

# K^{-1}
# fix this inversion!!
# inv_dyn_mat = np.linalg.inv(dyn_mat)
inv_dyn_mat = alternative_inversion(dyn_mat)

# Checking how far we are from equilibrium in this approximation 
displacements_dft = np.real(inv_dyn_mat @ dft_forces)

# Calculating displacements
f_tot = dft_forces + excited_forces
displacements = np.real(inv_dyn_mat @ f_tot)

# print(displacements)
# print(make_CM_disp_null(displacements, masses, atoms_species))

if CM_disp_null == True:
    displacements = make_CM_disp_null(displacements, masses, atoms_species)
    displacements_dft = make_CM_disp_null(displacements_dft, masses, atoms_species)

def write_displacements(displacements, arq_name):
    print(f"Writing {displacements} file")
    arq_out = open(arq_name, 'w')

    for iatom in range(Natoms):
        r = displacements[3*iatom:3*(iatom+1)]
        arq_out.write(f"{iatom+1}    {r[0]:.8f}     {r[1]:.8f}    {r[2]:.8f} \n")

    arq_out.close()
    
write_displacements(displacements, 'displacements_Fdft_Fexcited.dat')
write_displacements(displacements_dft, 'displacements_Fdft.dat')
estimate_energy_change(displacements, displacements_dft, excited_forces, dyn_mat)

print('Finished!')
