
#!/usr/bin/env python


# TODO
# 1 - checar derivadas do kernel
# 2 - checar se termos nao diagonais incluidos aqui estao certos
# 3 - Ler energias qp dos arquivos bandstructures.dat

# Tabela de dados pra ser escrita
# Diag = A, OffDiag = B (David Thesis)
# DKernel = C (IBL original paper)
# DModKernel = D (IBL original paper, after David's correction)
# A | B | C | D | A + B + D | A + C

import numpy as np
import h5py
from excited_forces_m import *
import time

start_time = time.clock_gettime(0)

TOL_DEG = 1e-5
Ry2eV = 13.6056980659
bohr2A = 0.529177249

# Dumb default parameters
Nkpoints = 1
Nvbnds = 1
Ncbnds = 1
Nval = 1
Nat = 1
iexc = 1
alat = 10 # FIXME: read it from input file or other source (maybe read volume instead)
# files and paths to be opened 
eqp_file = 'eqp1.dat'
exciton_dir = './'
el_ph_dir = './'
dyn_file = 'dyn'
kernel_file = 'bsemat.h5'
# conditionals
just_real = False
calc_modes_basis = False
calc_IBL_way = True
write_DKernel = False
report_RPA_data = False
just_RPA_diag = False
Calculate_Kernel = False

def read_input(input_file):

    # getting necessary info

    global alat
    global Nkpoints, Nvbnds, Ncbnds, Nval
    global Nat, iexc
    global eqp_file, exciton_dir, el_ph_dir
    global dyn_file, kernel_file
    global just_real, calc_modes_basis
    global calc_IBL_way

    arq_in = open(input_file)

    for line in arq_in:
        linha = line.split()
        if len(linha) >= 2:
            if linha[0] == 'Nkpoints':
                Nkpoints = int(linha[1])
            if linha[0] == 'Nvbnds':
                Nvbnds = int(linha[1])
            if linha[0] == 'Ncbnds':
                Ncbnds = int(linha[1])
            if linha[0] == 'Nval':
                Nval = int(linha[1])
            if linha[0] == 'Nat':
                Nat = int(linha[1])
            if linha[0] == 'iexc':
                iexc = int(linha[1])
            if linha[0] == 'eqp_file':
                eqp_file = linha[1]
            if linha[0] == 'exciton_dir':
                exciton_dir = linha[1]
            if linha[0] == 'el_ph_dir':
                el_ph_dir = linha[1]
            if linha[0] == 'just_real':
                if linha[1] == 'True':
                    just_real = True
            if linha[0] == 'dyn_file':
                dyn_file = linha[1]
            if linha[0] == 'kernel_file':
                kernel_file = linha[1]
            if linha[0] == 'calc_modes_basis':
                if linha[1] == 'True':
                    calc_modes_basis = True
            if linha[0] == 'calc_IBL_way':
                if linha[1] == 'True':
                    calc_IBL_way = True
            if linha[0] == 'alat':
                alat = float(linha[1])

    arq_in.close()


read_input('forces.inp')
exciton_file = exciton_dir+'/Avck_'+str(iexc)

Nmodes = Nat*3

print(f'alat = {alat}')
Vol = (alat/bohr2A)**3
Kernel_bgw_factor = Vol/(8*np.pi)

if Nvbnds > Nval:
    print('Warning! Nvbnds > Nval. Reseting Nvbnds to Nval')
    Nvbnds = Nval


print('\n---- Parameters -----\n')
print('Number of atoms : ' + str(Nat))
print('Number of modes (3*Nat) : ', Nmodes)
print('Nvbnds = '+str(Nvbnds) + ', Ncbnds = '+str(Ncbnds))
print('Valence band : ', Nval)
print('Nkpts = '+str(Nkpoints))
print('Exciton index to be read : '+str(iexc))
if calc_IBL_way == True:
    print('Calculating derivatives of Kernel using Ismail-Beigi and Louie\'s paper approach')
print('\n---------------------\n\n')


params_calc = Nkpoints, Ncbnds, Nvbnds, Nval, Nmodes

# Variables 

Shape = (Nmodes, Nkpoints, Ncbnds, Nvbnds)
Shape2 = (Nmodes, Nkpoints, Ncbnds, Nvbnds, Nkpoints, Ncbnds, Nvbnds)

DKinect          = np.zeros(Shape2, dtype=np.complex64) 

DKinect_diag     = np.zeros(Shape, dtype=np.complex64)
DKinect_offdiag  = np.zeros(Shape, dtype=np.complex64)

Sum_DKinect_diag            = np.zeros((Nmodes), dtype=np.complex64)
Sum_DKinect                 = np.zeros((Nmodes), dtype=np.complex64)

if Calculate_Kernel == True:
    Sum_DKernel            = np.zeros((Nmodes), dtype=np.complex64)
    Sum_DKernel_IBL        = np.zeros((Nmodes), dtype=np.complex64)

Forces_disp           = np.zeros((Nmodes), dtype=np.complex64)

Forces_modes          = np.zeros((Nmodes), dtype=np.complex64)

############ Getting info from files #############

# getting info from eqp.dat
Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, Nkpoints, Nvbnds, Ncbnds, Nval)

# Getting exciton info
Akcv, exc_energy = get_exciton_info(exciton_file, Nkpoints, Nvbnds, Ncbnds)
#Akcv, exc_energy = get_hdf5_exciton_info(exciton_dir+'/eigenvectors.h5', iexc)


print("Max real value of Akcv: ", np.max(np.real(Akcv)))
print("Max imag value of Akcv: ", np.max(np.imag(Akcv)))

if Calculate_Kernel == True:
    # # Getting kernel info
    Kx, Kd = get_kernel(kernel_file) 

    # # Must have same units of Eqp and Edft -> eV
    Kx =  - Kx * Ry2eV / Kernel_bgw_factor
    Kd =  - Kd * Ry2eV / Kernel_bgw_factor


# # Printing exciton energies

Mean_Kx, Mean_Kd, Mean_Ekin = 0.0, 0.0, 0.0


for ik1 in range(Nkpoints):
    for ic1 in range(Ncbnds):
        for iv1 in range(Nvbnds):
            Mean_Ekin += (Eqp_cond[ik1, ic1] - Eqp_val[ik1, iv1])*Akcv[ik1, ic1, iv1]*np.conj(Akcv[ik1, ic1, iv1])

if Calculate_Kernel == True:
    for ik1 in range(Nkpoints):
        for ic1 in range(Ncbnds):
            for iv1 in range(Nvbnds):
                for ik2 in range(Nkpoints):
                    for ic2 in range(Ncbnds):
                        for iv2 in range(Nvbnds):
                            Mean_Kx += np.conj(Akcv[ik1, ic1, iv1]) * Kx[ik2, ik1, ic2, ic1, iv2, iv1] * Akcv[ik2, ic2, iv2]
                            Mean_Kd += np.conj(Akcv[ik1, ic1, iv1]) * Kd[ik2, ik1, ic2, ic1, iv2, iv1] * Akcv[ik2, ic2, iv2]

print('Exciton energies (eV): ')
print('<KE> = ', Mean_Ekin)
print('Omega = ', exc_energy)
if Calculate_Kernel == True:
    print('<Kx> = ', Mean_Kx)
    print('<Kd> = ', Mean_Kd)
    print('DIFF ', exc_energy - (Mean_Ekin + Mean_Kd + Mean_Kx))

# get displacement patterns

iq = 0 # FIXME -> generalize for set of q points

Displacements, Nirreps = get_patterns2(el_ph_dir, iq, Nmodes, Nat)
elph = get_el_ph_coeffs(el_ph_dir, iq, Nirreps)
elph_cond, elph_val = filter_elph_coeffs(elph, params_calc)

########## Calculating stuff ############

print("Calculating matrix elements for forces calculations <cvk|dH/dx_mu|c'v'k'>")
print("    - Calculating RPA part")
# Creating auxialiry quantities

aux_diag = np.zeros(Shape, dtype=np.complex64)  # <ck|dV/du_mode|ck> - <vk|dV/du_mode|vk>
aux_offdiag = np.zeros(Shape, dtype=np.complex64)

aux_cond_matrix, aux_val_matrix = aux_matrix_elem(Nmodes, Nkpoints, Ncbnds, Nvbnds, elph_cond, elph_val, Edft_val, Edft_cond, Eqp_val, Eqp_cond, TOL_DEG)

DKinect = calc_Dkinect_matrix(params_calc, Akcv, aux_cond_matrix, aux_val_matrix, report_RPA_data, just_RPA_diag)


# Forces from Kernel derivatives
if Calculate_Kernel == True:

    EDFT = Edft_val, Edft_cond
    EQP = Eqp_val, Eqp_cond
    Params = Ncbnds, Nvbnds, Nkpoints, Nmodes
    ELPH = elph_cond, elph_val

    if calc_IBL_way == True:
        DKernel, DKernel_IBL = calc_deriv_Kernel(Kx+Kd, calc_IBL_way, EDFT, EQP, ELPH, TOL_DEG, Params, Akcv)
    else:
        DKernel = calc_deriv_Kernel(Kx+Kd, calc_IBL_way, EDFT, EQP, ELPH, TOL_DEG, Params, Akcv)

print("Calculating sums")

for imode in range(Nmodes):

    sum_temp = 0.0 + 0.0j
    for ik in range(Nkpoints):
        for ic in range(Ncbnds):
            for iv in range(Nvbnds):
                sum_temp += DKinect[imode, ik, ic, iv, ik, ic, iv]

    Sum_DKinect_diag[imode] = sum_temp

    # sum of off-diagonal part + sum of diagonal part
    Sum_DKinect[imode] = np.sum(DKinect[imode])

    Sum_DKernel[imode] = np.sum(DKernel[imode])
    if calc_IBL_way == True:
        Sum_DKernel_IBL[imode] = np.sum(DKernel_IBL[imode])


# Convert from Ry/bohr to eV/A. Minus sign comes from F=-dV/du
Sum_DKinect_diag = -Sum_DKinect_diag*Ry2eV/bohr2A
Sum_DKinect = -Sum_DKinect*Ry2eV/bohr2A
Sum_DKernel = -Sum_DKernel*Ry2eV/bohr2A
if calc_IBL_way == True:
    Sum_DKernel_IBL = -Sum_DKernel_IBL*Ry2eV/bohr2A

# Calculate forces cartesian basis

print("Calculating forces in cartesian basis")

F_cart_KE_IBL                       = np.zeros((Nat, 3), dtype=np.complex64)  # IBL just diag RPA
F_cart_KE_David                     = np.zeros((Nat, 3), dtype=np.complex64)  # david thesis - diag + offdiag from kinect part
F_cart_Kernel_IBL                   = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's paper 
F_cart_Kernel_IBL_correct           = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's with new kernel

for iatom in range(Nat):
    for imode in range(Nmodes):
        F_cart_KE_IBL[iatom] += Displacements[imode, iatom] * Sum_DKinect_diag[imode]
        F_cart_KE_David[iatom] += Displacements[imode, iatom] * Sum_DKinect[imode]
        F_cart_Kernel_IBL[iatom] += Displacements[imode, iatom] * (Sum_DKernel_IBL[imode] + Sum_DKinect_diag[imode])
        F_cart_Kernel_IBL_correct[iatom] += Displacements[imode, iatom] * (Sum_DKernel[imode] + Sum_DKinect_diag[imode])

#print('\n\n\n################# Forces (eV/A) in cartesian basis #####################')

DIRECTION = ['x', 'y', 'z']

arq_out = open('forces_cart.out', 'w')

print('\n\nForces (eV/ang)\n')
print('# Atom  dir  RPA_diag RPA_diag_offiag RPA_diag_Kernel RPA_diag_newKernel')
arq_out.write('# Atom  dir  RPA_diag RPA_diag_offiag RPA_diag_Kernel RPA_diag_newKernel\n')

for iatom in range(Nat):
    for idir in range(3):
        text =  str(iatom+1)+' '+DIRECTION[idir]+' '
        text += str(            F_cart_KE_IBL[iatom, idir])+' '
        text += str(          F_cart_KE_David[iatom, idir])+' '
        text += str(        F_cart_Kernel_IBL[iatom, idir])+' '
        text += str(F_cart_Kernel_IBL_correct[iatom, idir])
        print(text)
        arq_out.write(text+'\n')

arq_out.close()

end_time = time.clock_gettime(0)

print('\n\nCalculation finished!')
print(f'Total time: {round(end_time - start_time, 1)} (s)')
