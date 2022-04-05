
#!/usr/bin/env python


# TO DO
# 1 - incluir derivadas do kernel. ler arquivos hdf5
# 2 - checar se termos nao diagonais incluidos aqui estao certos
# 3 - criar modulo pra esse codigo aqui
# 4 - ler dados dos calculos dfpt usando modulo pra ler arquivos .xml
# 5 - ler dados de excitons lendo arquivos binarios
# 6 - Estou usando todas as bandas de conducao calculadas a nivel DFT. Mudar para escolher um certo numero disso, verificando em cada etapa que bandas estao disponiveis


# Tabela de dados pra ser escrita
# Diag = A, OffDiag = B (David Thesis)
# DKernel = C (IBL original paper)
# DModKernel = D (IBL original paper, after David's correction)
# A | B | C | D | A + B + D | A + C

import numpy as np
import h5py
from excited_forces_m import *

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
write_DKernel = True

def read_input(input_file):

    # getting necessary info

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

    arq_in.close()


read_input('forces.inp')
exciton_file = exciton_dir+'/Avck_'+str(iexc)

Nmodes = Nat*3

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

a = 15/bohr2A # FIXME: read it from input file or other source (maybe read volume instead)
Vol = a**3
Kernel_bgw_factor = Vol/(8*np.pi)

# Variables 

Shape = (Nmodes, Nkpoints, Ncbnds, Nvbnds)
Shape2 = (Nmodes, Nkpoints, Ncbnds, Nvbnds, Nkpoints, Ncbnds, Nvbnds)

DKinect          = np.zeros(Shape2, dtype=np.complex64) 

DKinect_diag     = np.zeros(Shape, dtype=np.complex64)
DKinect_offdiag  = np.zeros(Shape, dtype=np.complex64)
#DKernel          = np.zeros(Shape2, dtype=np.complex64)
#DKernel_IBL      = np.zeros(Shape2, dtype=np.complex64)

Sum_DKinect_diag            = np.zeros((Nmodes), dtype=np.complex64)
Sum_DKinect                 = np.zeros((Nmodes), dtype=np.complex64)
#Sum_DKernel            = np.zeros((Nmodes), dtype=np.complex64)
#Sum_DKernel_IBL        = np.zeros((Nmodes), dtype=np.complex64)

Forces_disp           = np.zeros((Nmodes), dtype=np.complex64)

Forces_modes          = np.zeros((Nmodes), dtype=np.complex64)

############ Getting info from files #############

# getting info from eqp.dat
Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, Nkpoints, Nvbnds, Ncbnds, Nval)

# Getting exciton info
Akcv, exc_energy = get_exciton_info(exciton_file, Nkpoints, Nvbnds, Ncbnds)

# Getting kernel info
#Kx, Kd = get_kernel(kernel_file) 

# Must have same units of Eqp and Edft -> eV
#Kx = Kx * Ry2eV / Kernel_bgw_factor
#Kd = Kd * Ry2eV / Kernel_bgw_factor

# Printing exciton energies

#Mean_Kx, Mean_Kd, Mean_Ekin = 0.0, 0.0, 0.0

#for ik in range(Nkpoints):
#    for ic1 in range(Ncbnds):
#        for iv1 in range(Nvbnds):
#            Mean_Ekin += (Eqp_cond[ik, ic1] - Eqp_val[ik, iv1])*abs(Akcv[ik, ic1, iv1])**2
#            for ic2 in range(Ncbnds):
#                for iv2 in range(Nvbnds):
#                    Mean_Kx += np.conj(Akcv[ik, ic1, iv1]) * Kx[ik, ik, ic1, ic2, iv1, iv2] * Akcv[ik, ic2, iv2]
#                    Mean_Kd += np.conj(Akcv[ik, ic1, iv1]) * Kd[ik, ik, ic1, ic2, iv1, iv2] * Akcv[ik, ic2, iv2]

#print('Exciton energies (eV): ')
#print('<Kx> = ', Mean_Kx)
#print('<Kd> = ', Mean_Kd)
#print('<KE> = ', Mean_Ekin)
#print('Omega = ', exc_energy)

# get displacement patterns

iq = 0 # FIXME -> generalize for set of q points
#Displacements, Nirreps, Perts = get_patterns(el_ph_dir, iq, Nmodes, Nat)
#elph_val, elph_cond, elph_aux = get_el_ph_coeffs(el_ph_dir, iq, Nirreps, Perts, Nmodes, Nkpoints, Ncbnds, Nvbnds, Nval)

# print('Displacements: ', Displacements)


Displacements, Nirreps, Perts = get_patterns2(el_ph_dir, iq, Nmodes, Nat)
elph_aux, elph_cond, elph_val = get_el_ph_coeffs2(el_ph_dir, iq, Nirreps, params_calc)
#elph_cond, elph_val = filter_elph_coeffs(elph_aux, Ncbnds, Nvbnds, Nkpoints, Nmodes, Nval)

print("Max real value of <c|dH|c'> (eV/A): ", np.max(np.real(elph_cond)))
print("Max imag value of <c|dH|c'> (eV/A): ", np.max(np.imag(elph_cond)))
print("Max real value of <v|dH|v'> (eV/A): ", np.max(np.real(elph_val)))
print("Max imag value of <v|dH|v'> (eV/A): ", np.max(np.imag(elph_val)))
########## Calculating stuff ############

# Creating auxialiry quantities

# Derivatives of diagonal kinect part

aux_diag = np.zeros(Shape, dtype=np.complex64)  # <ck|dV/du_mode|ck> - <vk|dV/du_mode|vk>
aux_offdiag = np.zeros(Shape, dtype=np.complex64)

# for imode in range(Nmodes):
#     for ik in range(Nkpoints):
#         for ic in range(Ncbnds):
#             for iv in range(Nvbnds):
#                 Fcvk_diag, Fcvk_offdiag = calculate_Fcvk(Ncbnds, Nvbnds, Akcv, Edft_cond, Edft_val, Eqp_cond, Eqp_val, elph_cond, elph_val, imode, ik, ic, iv, TOL_DEG)
#                 #print('DEBUG', ik, ic, iv, Fcvk_diag, Fcvk_offdiag)
#                 aux_diag[imode, ik, ic, iv] = Fcvk_diag
#                 aux_offdiag[imode, ik, ic, iv] = Fcvk_offdiag

aux_cond_matrix, aux_val_matrix = aux_matrix_elem(Nmodes, Nkpoints, Ncbnds, Nvbnds, elph_cond, elph_val, Edft_val, Edft_cond, Eqp_val, Eqp_cond, TOL_DEG)

for imode in range(Nmodes):
    for ik in range(Nkpoints):
        for ic1 in range(Ncbnds):
            for ic2 in range(Ncbnds):
                for iv1 in range(Nvbnds):
                    for iv2 in range(Nvbnds):

                        temp = calc_Dkinect_matrix_elem(Akcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2)
                        DKinect[imode, ik, ic1, iv1, ik, ic2, iv2] = temp


# # Compute diag elements - kinetic part
# for imode in range(Nmodes):
#     DKinect_diag[imode] = np.conj(Akcv)*aux_diag[imode]

# # Compute offdiag elements - kinetic part 
# for imode in range(Nmodes):
#     DKinect_offdiag[imode] = np.conj(Akcv)*aux_offdiag[imode]

# Forces from Kernel derivatives

# EDFT = Edft_val, Edft_cond
# EQP = Eqp_val, Eqp_cond
# Nparams = Ncbnds, Nvbnds, Nkpoints
# ELPH = elph_cond, elph_val

# for imode in range(Nmodes):
#     for ik1 in range(Nkpoints):
#         for ic1 in range(Ncbnds):
#             for iv1 in range(Nvbnds):

#                 A_bra = np.conj(Akcv[ik1, ic1, iv1])

#                 for ik2 in range(Nkpoints):
#                     for ic2 in range(Ncbnds):
#                         for iv2 in range(Nvbnds):

#                             A_ket = Akcv[ik2, ic2, iv2]

#                             indexes = ik1, ik2, iv1, iv2, ic1, ic2, imode
#                             dK = calc_DKernel(indexes, Kx + Kd, calc_IBL_way, EDFT, EQP, ELPH, Nparams, TOL_DEG)

#                             if calc_IBL_way == False:
#                                 DKernel[imode, ik1, ic1, iv1, ik2, ic2, iv2] = A_bra * dK * A_ket
#                             else:
#                                 DKernel[imode, ik1, ic1, iv1, ik2, ic2, iv2] = A_bra * dK[0] * A_ket
#                                 DKernel_IBL[imode, ik1, ic1, iv1, ik2, ic2, iv2] = A_bra * dK[1] * A_ket


# Sums
# for imode in range(Nmodes):
#     Sum_DKinect_diag[imode] = np.sum(DKinect_diag[imode])
#     Sum_DKinect_offdiag[imode] = np.sum(DKinect_offdiag[imode])
#    Sum_DKernel[imode] = np.sum(DKernel[imode])
#    if calc_IBL_way == True:
#        Sum_DKernel_IBL[imode] = np.sum(DKernel_IBL[imode])

for imode in range(Nmodes):

    sum_temp = 0.0 + 1.0j
    for ik in range(Nkpoints):
        for ic in range(Ncbnds):
            for iv in range(Nvbnds):
                sum_temp += DKinect[imode, ik, ic, iv, ik, ic, iv]

    Sum_DKinect_diag[imode] = sum_temp

    # sum of off-diagonal part + sum of diagonal part
    Sum_DKinect[imode] = np.sum(DKinect[imode])


# Convert to eV/A. Minus sign comes from F=-dV/du
Sum_DKinect_diag = -Sum_DKinect_diag*Ry2eV/bohr2A
Sum_DKinect = -Sum_DKinect*Ry2eV/bohr2A
#Sum_DKernel = -Sum_DKernel*Ry2eV/bohr2A
#if calc_IBL_way == True:
#    Sum_DKernel_IBL = -Sum_DKernel_IBL*Ry2eV/bohr2A

# Calculate forces cartesian basis

F_cart_KE_IBL                       = np.zeros((Nat, 3), dtype=np.complex64)  # david thesis - diag + offdiag from kinect part
F_cart_KE_David                     = np.zeros((Nat, 3), dtype=np.complex64)  # david thesis - diag + offdiag from kinect part + derivative of Kernel (corrected)
#F_cart_Kernel_IBL                   = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's paper 
#F_cart_Kernel_IBL_correct           = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's paper with dK = 0
#F_cart_Kernel_IBL_correct_extended  = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's paper with new correct dK 

for iatom in range(Nat):
    for imode in range(Nmodes):
        F_cart_KE_IBL[iatom] += Displacements[imode, iatom] * Sum_DKinect_diag[imode]
        F_cart_KE_David[iatom] += Displacements[imode, iatom] * (Sum_DKinect[imode])
#        F_cart_Kernel_IBL[iatom] += Displacements[imode, iatom] * Sum_DKernel_IBL[imode] 
#        F_cart_Kernel_IBL_correct[iatom] += Displacements[imode, iatom] * Sum_DKernel[imode]

#print('\n\n\n################# Forces (eV/A) in cartesian basis #####################')

DIRECTION = ['x', 'y', 'z']

arq_out = open('forces_cart.out', 'w')

print('\n\nForces (eV/ang)\n')
print('# Atom  dir  KE_IBL KE_David Kernel_IBL Kernel_IBL_correct')
arq_out.write('# Atom  dir  KE_IBL KE_David Kernel_IBL Kernel_IBL_correct\n')

for iatom in range(Nat):
    for idir in range(3):
        text =  str(iatom+1)+' '+DIRECTION[idir]+' '
        text += str(F_cart_KE_IBL[iatom][idir])+' '
        text += str(F_cart_KE_David[iatom][idir])+' '
#        text += str(F_cart_Kernel_IBL[iatom][idir])+' '
#        text += str(F_cart_Kernel_IBL_correct[iatom][idir])
        print(text)
        arq_out.write(text+'\n')

arq_out.close()
