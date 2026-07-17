
'''

 Code usage:
 python cart2ph_eigvec.py forces_cart.out displacements.axsf

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


OLD OLD OLD OLD

THE DISPLACEMENTS IN THOSE FILES ARE IN UNITS OF DISPLACEMENT / SQRT(MASS)

#    The eigvecs file is the output from dynmat.x (QE) that is done after the PH calculation.
#    It is recommended to inclulde ASR in this calculation.
#    This code assumes displacements are real values and eigvecs are normalized to 1.
#    Just calculate phonons for q = (0,0,0)!
#    This file looks something like this

# ####################################################################################

#         diagonalizing the dynamical matrix ...

#  q =       0.0000      0.0000      0.0000
#  **************************************************************************
#      freq (    1) =      -0.000001 [THz] =      -0.000021 [cm-1]
#  (  0.076991   0.000000     0.076977   0.000000    -0.086716   0.000000   )
#  (  0.083142   0.000000     0.083127   0.000000    -0.093645   0.000000   )
#  (  0.022303   0.000000     0.022299   0.000000    -0.025121   0.000000   )
#      freq (    2) =       0.000000 [THz] =       0.000008 [cm-1]
#  (  0.101322   0.000000    -0.095277   0.000000     0.005383   0.000000   )
#  (  0.109418   0.000000    -0.102889   0.000000     0.005813   0.000000   )
 
# ####################################################################################

NEW 

Now we read the normal modes displacements from the dynmat.axsf file (produced by dynmat.x program from QE).
In this file the displacements are given in lenght untis (insted of lenght / sqrt(mass) that are eigvecs from the dynamical matrix)
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
   C     -0.60956  -0.00106   2.86751   0.01597   0.01597  -0.01799  # atom_name x y z fx fy fz
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

print('File with forces in cartesian basis: ', forces_cart_file)
print('File with phonon eigenvectors: ', eigvecs_file)

# reading forces file in cartesian basis
print('Reading forces file in cartesian basis')
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
            try:
               forces_cart[ii-2].append(float(line_split[ii]))
            except:
               forces_cart[ii-2].append(complex(line_split[ii]))

arq_forces.close()

# transforming each force in array
for i_force in range(len(forces_cart)):
   forces_cart[i_force] = np.array(forces_cart[i_force])

print('Finished reading forces file')

# # reading eigvecs
# print('Reading eigvecs file')
# arq_eigvecs = open(eigvecs_file)

# eigvecs = []
# freqs = []
# for line in arq_eigvecs:
#    line_split = line.split()
#    if len(line_split) > 0:
#       if line_split[0] == 'freq':
#          freqs.append(float(line_split[-2]))
#          eigvecs.append([])
#       if line_split[0] == '(':
#          for i_dir in [1,3,5]:
#             eigvecs[-1].append(float(line_split[i_dir]))

# for i_eigvec in range(len(eigvecs)):
#    eigvecs[i_eigvec] = np.array(eigvecs[i_eigvec])
#    # print('Norm = ', np.dot(eigvecs[i_eigvec], eigvecs[i_eigvec]))
   
# arq_eigvecs.close()
# print('Finished reading eigvecs file')

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
for i_eigvec in range(len(displacements)):
   displacements[i_eigvec] = displacements[i_eigvec] / np.sqrt(np.dot(displacements[i_eigvec], displacements[i_eigvec]))

# calculating forces in ph basis and report this data

print('Calculatinig forces in eigvecs basis')
print('Writing data in forces_phonons_basis.out \n\n')

output = open('forces_phonons_basis.out', 'w')

output.write('# i_eigvec ')
for name in NAMES:
   output.write(name + '    ')
output.write('\n')

for i_eigvec in range(len(displacements)):
   line_output = f'{i_eigvec+1}    '
   for i_force in range(len(forces_cart)):
      f_ph_basis = np.dot(displacements[i_eigvec], forces_cart[i_force])
      line_output += f'{f_ph_basis}      '
   line_output += '\n'
   output.write(line_output)

output.close()
print('Done!')
