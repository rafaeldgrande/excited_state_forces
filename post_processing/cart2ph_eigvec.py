
'''

 Code usage:
 python cart2ph_eigvec.py forces_cart.out displacements.axsf forces_phonons_basis.out

 Output: forces_phonons_basis.out -> forces in phonon basis


   This code reads the forces_cart.out file with the excited state forces in CARTESIAN basis
   and then converts it to a phonon basis obtained from eigvecs file.
   Assumes that the forces are real values.
   
   forces_cart.out is the output from the excited_state_forces code and contains the excited 
   state forces in each direction for each atom. It looks something like this

####################################################################################
   # Atom  dir  RPA_diag RPA_diag_offiag 
   1 x 0.10997382992166772 0.06055312291498007
   1 y -0.00019108915877979644 0.005851683557188331
   1 z -0.12640693104100015 -0.10717424014630982
   2 x 0.03169959177566452 0.08302436544336893
   2 y 4.5571266927958986e-05 0.0858895738534057
   2 z 0.19798424636217724 0.1936890925371606
####################################################################################


We read the phonon modes displacements from the dynmat.axsf file (produced by dynmat.x program from QE).
In this file the displacements are given in lenght untis (insted of lenght / sqrt(mass)
Notice that those displacements ARE NOT perpendicular to each other, while the eigenvectors from the dynamical matrix ARE orthogonal to each other.
The file has the following structure:

ANIMSTEPS   36 -> number of modes
CRYSTAL
PRIMVEC
    6.165941614   -0.000492887   -0.005888717
    0.001024842    6.113176002   -0.000179232
    0.131058401   -0.000553227    6.258428284
PRIMCOORD    1
     12   1  
   C     -0.60956  -0.00106   2.86751   0.01597   0.01597  -0.01799  # atom_name x y z disp_x disp_y disp_z
   N      0.61374   0.00013  -2.81849   0.01597   0.01597  -0.01799
   H     -0.53433  -0.00192   1.76995   0.01597   0.01597  -0.01799
   H     -1.27398   0.90188  -3.05512   0.01597   0.01597  -0.01799
   H     -1.27368  -0.90255  -3.05372   0.01597   0.01597  -0.01799
   H      1.15868   0.84559  -3.11247   0.01597   0.01597  -0.01799
   H      1.15893  -0.84556  -3.11146   0.01597   0.01597  -0.01799
   H      0.61935   0.00077  -1.77562   0.01597   0.01597  -0.01799
   Pb     2.74588   3.05610  -0.16654   0.01597   0.01597  -0.01799
   I      2.64123   3.05593   2.95346   0.01597   0.01597  -0.01799
   I      2.44579  -0.00050   0.19764   0.01597   0.01597  -0.01799
   I     -0.40025   3.05622  -0.54624   0.01597   0.01597  -0.01799
PRIMCOORD    2
     12   1
   C     -0.60956  -0.00106   2.86751   0.02101  -0.01976   0.00112
   N      0.61374   0.00013  -2.81849   0.02101  -0.01976   0.00112
   H     -0.53433  -0.00192   1.76995   0.02101  -0.01976   0.00112
   H     -1.27398   0.90188  -3.05512   0.02101  -0.01976   0.00112
   H     -1.27368  -0.90255  -3.05372   0.02101  -0.01976   0.00112
...
    '''

import sys
import numpy as np

# taking inputs
forces_cart_file = sys.argv[1]
eigvecs_file = sys.argv[2]
output_file = sys.argv[3]

NAMES = ['RPA_diag', 'RPA_diag_offiag', 'RPA_diag+Kernel']

print('File with forces in cartesian basis: ', forces_cart_file)
print('File with phonon eigenvectors: ', eigvecs_file)

# reading forces file in cartesian basis
print('Reading forces file in cartesian basis')               
forces_cart = np.loadtxt(forces_cart_file, usecols=[2, 3, 4], dtype=complex)               
print('Finished reading forces file')


# getting phonon displacements
print('Reading eigvecs file')
arq_eigvecs = open(eigvecs_file)

displacements = []

for line in arq_eigvecs:
   line_split = line.split()
   if len(line_split) == 7:
      dx = float(line_split[4])
      dy = float(line_split[5]) 
      dz = float(line_split[6])
      displacements[-1].append(dx)
      displacements[-1].append(dy)
      displacements[-1].append(dz)
   elif len(line_split) == 2:
      if line_split[0] == 'PRIMCOORD':
         displacements.append([])

# converting it to array         
displacements = np.array(displacements)

# those eigvecs are not normalized, then we have to do it
print('Normalizing displacements')
for i_eigvec in range(len(displacements)):
   displacements[i_eigvec] = displacements[i_eigvec] / np.linalg.norm(displacements[i_eigvec])

# calculating forces in ph basis and report this data
print('Calculatinig forces in eigvecs basis')
print('Writing data in forces_phonons_basis.out \n\n')

COL_WIDTH = 30
with open(output_file, 'w') as output:
   output.write(f'# {"i_eigvec":>10s}')
   for name in NAMES:
      output.write(f'{name:>{COL_WIDTH}s}')
   output.write('\n')

   for i_eigvec in range(len(displacements)):
      line_output = f'  {i_eigvec+1:>10d}'
      for i_force in range(len(NAMES)):
         f_ph_basis = np.dot(displacements[i_eigvec], forces_cart[:, i_force])
         cplx_str = f'{f_ph_basis.real:.8f}{f_ph_basis.imag:+.8f}j'
         line_output += f'{cplx_str:>{COL_WIDTH}s}'
      line_output += '\n'
      output.write(line_output)
print('Done!')
