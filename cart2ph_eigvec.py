
'''This code reads the forces_cart.out file with the excited state forces in CARTESIAN basis
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


   The eigvecs file is the output from dynmat.x (QE) that is done after the PH calculation.
   It is recommended to inclulde ASR in this calculation.
   This code assumes displacements are real values and eigvecs are normalized to 1.
   Just calculate phonons for q = (0,0,0)!
   This file looks something like this

####################################################################################

        diagonalizing the dynamical matrix ...

 q =       0.0000      0.0000      0.0000
 **************************************************************************
     freq (    1) =      -0.000001 [THz] =      -0.000021 [cm-1]
 (  0.076991   0.000000     0.076977   0.000000    -0.086716   0.000000   )
 (  0.083142   0.000000     0.083127   0.000000    -0.093645   0.000000   )
 (  0.022303   0.000000     0.022299   0.000000    -0.025121   0.000000   )
     freq (    2) =       0.000000 [THz] =       0.000008 [cm-1]
 (  0.101322   0.000000    -0.095277   0.000000     0.005383   0.000000   )
 (  0.109418   0.000000    -0.102889   0.000000     0.005813   0.000000   )
 
####################################################################################

 Code usage:
 python cart2ph_eigvec.py forces_cart.out eigvecs

 Output: forces_phonons_basis.out -> forces in phonon basis

    '''

import sys
import numpy as np

# taking inputs
forces_cart_file = sys.argv[1]
eigvecs_file = sys.argv[2]

print('File with forces in cartesian basis: ', forces_cart_file)
print('File with phonon eigenvectors: ', eigvecs_file)

# reading forces file

print('Reading forces file')
arq_forces = open(forces_cart_file)

forces_cart = []
for line in arq_forces:
   line_split = line.split()
   if len(line_split) > 0:
      if line_split[0] == '#':  # -> # Atom  dir  RPA_diag RPA_diag_offiag Kernel Kernel_mod
         NAMES = line_split[3:]
         for i_name in range(len(NAMES)):
            forces_cart.append([])
      else:
         for ii in range(2, len(line_split)):
            forces_cart[ii-2].append(float(line_split[ii]))

arq_forces.close()

# transforming each force in array
for i_force in range(len(forces_cart)):
   forces_cart[i_force] = np.array(forces_cart[i_force])

print('Finished reading forces file')

# reading eigvecs

print('Reading eigvecs file')
arq_eigvecs = open(eigvecs_file)

eigvecs = []
freqs = []
for line in arq_eigvecs:
   line_split = line.split()
   if len(line_split) > 0:
      if line_split[0] == 'freq':
         freqs.append(float(line_split[-2]))
         eigvecs.append([])
      if line_split[0] == '(':
         for i_dir in [1,3,5]:
            eigvecs[-1].append(float(line_split[i_dir]))

for i_eigvec in range(len(eigvecs)):
   eigvecs[i_eigvec] = np.array(eigvecs[i_eigvec])
   # print('Norm = ', np.dot(eigvecs[i_eigvec], eigvecs[i_eigvec]))


arq_eigvecs.close()
print('Finished reading eigvecs file')

# calculating forces in ph basis and report this data

print('Calculatinig forces in eigvecs basis')
print('Writing data in forces_phonons_basis.out \n\n')

output = open('forces_phonons_basis.out', 'w')

output.write('# i_eigvec freq ')
for name in NAMES:
   output.write(name + '    ')
output.write('\n')

for i_eigvec in range(len(eigvecs)):
   line_output = f'{i_eigvec+1}    {freqs[i_eigvec]}    '
   for i_force in range(len(forces_cart)):
      f_ph_basis = np.dot(eigvecs[i_eigvec], forces_cart[i_force])
      line_output += f'{f_ph_basis}      '
   line_output += '\n'
   output.write(line_output)

output.close()
print('Done!')