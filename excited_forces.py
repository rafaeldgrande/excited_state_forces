
# FIRST MESSAGE

from datetime import datetime
# datetime object containing current date and time
now = datetime.now()

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("\n\nExecution date: ", dt_string)

print('\n\n*************************************************************')
print('Excited state forces code')
print('Developed by Rafael Del Grande and David Strubbe')
print('Contact: rdelgrande@ucmerced.edu')
print('*************************************************************')


# Importing modules
import numpy as np
import time
import tracemalloc # track ram usage
tracemalloc.start()


# My modules
from excited_forces_m import *
from qe_interface_m import *
from bgw_interface_m import *
from excited_forces_config import *


# Report functions
def report_time(start_time):
    end_time_func = time.clock_gettime(0)
    text = f'{(end_time_func - start_time)/60:.2f} min'
    return text

def report_ram():
    temp_ram = tracemalloc.get_traced_memory()[0] / 1024**2
    max_temp_ram = tracemalloc.get_traced_memory()[1] / 1024**2

    print('\n\n############### RAM REPORT #################')
    print(f'RAM used now: {temp_ram:.2f} MB')
    print(f'Max RAM used until now: {max_temp_ram:.2f} MB')
    print('############################################\n\n')


start_time = time.clock_gettime(0)

############ Getting info from files #############

class Parameters_BSE:

    def __init__(self, Nkpoints_BSE, Kpoints_BSE, Ncbnds, Nvbnds, Nval):
        self.Nkpoints_BSE   = Nkpoints_BSE
        self.Kpoints_BSE    = Kpoints_BSE
        self.Ncbnds         = Ncbnds
        self.Nvbnds         = Nvbnds
        self.Nval           = Nval

class Parameters_MF:

    def __init__(self, Nat, atomic_pos, cell_vecs, cell_vol, alat):
        self.Nat         = Nat
        self.atomic_pos  = atomic_pos
        self.cell_vecs   = cell_vecs
        self.cell_vol    = cell_vol
        self.alat        = alat
        self.Nmodes      = 3 * Nat
    

class Parameters_ELPH:

    def __init__(self, Nkpoints_DPFT, Kpoints_DFPT):
        self.Nkpoints_DFPT = Nkpoints_DPFT
        self.Kpoints_DFPT  = Kpoints_DFPT


# Getting exciton info and BSE and MF parameters

Akcv, Omega, Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval = get_hdf5_exciton_info(exciton_file, iexc)

BSE_params = Parameters_BSE(Nkpoints_BSE, Kpoints_BSE, Ncbnds, Nvbnds, Nval)
MF_params  = Parameters_MF(Nat, atomic_pos, cell_vecs, cell_vol, alat)

Nmodes = 3 * Nat


print("    Max real value of Akcv: ", np.max(np.real(Akcv)))
print("    Max imag value of Akcv: ", np.max(np.imag(Akcv)))
print('\n\n')

# getting info from eqp.dat
Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, BSE_params)

if Calculate_Kernel == True:
    # # Getting kernel info
    Kx, Kd = get_kernel(kernel_file) 

    # # Must have same units of Eqp and Edft -> eV
    Kx =  - Kx * Ry2eV / Kernel_bgw_factor
    Kd =  - Kd * Ry2eV / Kernel_bgw_factor


# get displacement patterns

iq = 0 # FIXME -> generalize for set of q points

Displacements, Nirreps = get_patterns2(iq, MF_params)
elph = get_el_ph_coeffs(iq, Nirreps)

if acoutic_sum_rule == True:
    print('\n\n')
    elph = impose_ASR(elph, Displacements, MF_params)

elph_cond, elph_val = filter_elph_coeffs(elph, MF_params, BSE_params)

# report_ram()
# print('DELETING ELPH')
del elph
report_ram()

# # Printing exciton energies

# TODO -> just create those matrices when they are necessary, then erase them when finished

Shape = (MF_params.Nmodes, Nkpoints_BSE, Ncbnds, Nvbnds)
Shape2 = (MF_params.Nmodes, Nkpoints_BSE, Ncbnds, Nvbnds, Nkpoints_BSE, Ncbnds, Nvbnds)

print('SHAPE', Shape)

DKinect          = np.zeros(Shape2, dtype=np.complex64) 

Sum_DKinect_diag            = np.zeros((MF_params.Nmodes), dtype=np.complex64)
Sum_DKinect                 = np.zeros((MF_params.Nmodes), dtype=np.complex64)

if Calculate_Kernel == True:
    Sum_DKernel            = np.zeros((MF_params.Nmodes), dtype=np.complex64)
    Sum_DKernel_IBL        = np.zeros((MF_params.Nmodes), dtype=np.complex64)

Forces_disp           = np.zeros((MF_params.Nmodes), dtype=np.complex64)

Forces_modes          = np.zeros((MF_params.Nmodes), dtype=np.complex64)


Mean_Kx, Mean_Kd, Mean_Ekin = 0.0, 0.0, 0.0


for ik1 in range(Nkpoints_BSE):
    for ic1 in range(Ncbnds):
        for iv1 in range(Nvbnds):
            Mean_Ekin += (Eqp_cond[ik1, ic1] - Eqp_val[ik1, iv1])*Akcv[ik1, ic1, iv1]*np.conj(Akcv[ik1, ic1, iv1])

if Calculate_Kernel == True:
    for ik1 in range(Nkpoints_BSE):
        for ic1 in range(Ncbnds):
            for iv1 in range(Nvbnds):
                for ik2 in range(Nkpoints_BSE):
                    for ic2 in range(Ncbnds):
                        for iv2 in range(Nvbnds):
                            Mean_Kx += np.conj(Akcv[ik1, ic1, iv1]) * Kx[ik2, ik1, ic2, ic1, iv2, iv1] * Akcv[ik2, ic2, iv2]
                            Mean_Kd += np.conj(Akcv[ik1, ic1, iv1]) * Kd[ik2, ik1, ic2, ic1, iv2, iv1] * Akcv[ik2, ic2, iv2]

print('Exciton energies (eV): ')
print('<KE> = ', Mean_Ekin)
print('Omega = ', Omega)
if Calculate_Kernel == True:
    print('<Kx> = ', Mean_Kx)
    print('<Kd> = ', Mean_Kd)
    print('DIFF ', Omega - (Mean_Ekin + Mean_Kd + Mean_Kx))

report_ram()

########## Calculating stuff ############

print("Calculating matrix elements for forces calculations <cvk|dH/dx_mu|c'v'k'>")
print("    - Calculating RPA part")
# Creating auxialiry quantities

aux_diag = np.zeros(Shape, dtype=np.complex64)  # <ck|dV/du_mode|ck> - <vk|dV/du_mode|vk>
aux_offdiag = np.zeros(Shape, dtype=np.complex64)

aux_cond_matrix, aux_val_matrix = aux_matrix_elem(elph_cond, elph_val, Eqp_val, Eqp_cond, Edft_val, Edft_cond, MF_params, BSE_params)

# Calculating matrix elements F_cvkc'v'k'
DKinect = calc_Dkinect_matrix(Akcv, aux_cond_matrix, aux_val_matrix, MF_params, BSE_params)

# Kernel derivatives
if Calculate_Kernel == True:

    EDFT = Edft_val, Edft_cond
    EQP = Eqp_val, Eqp_cond
    ELPH = elph_cond, elph_val

    if calc_IBL_way == True:
        DKernel, DKernel_IBL = calc_deriv_Kernel(Kx+Kd, EDFT, EQP, ELPH, Akcv)
    else:
        DKernel = calc_deriv_Kernel(Kx+Kd, EDFT, EQP, ELPH, Akcv)

print("Calculating sums")

for imode in range(Nmodes):

    sum_temp = 0.0 + 0.0j
    for ik in range(Nkpoints_BSE):
        for ic in range(Ncbnds):
            for iv in range(Nvbnds):
                sum_temp += DKinect[imode, ik, ic, iv, ik, ic, iv]

    Sum_DKinect_diag[imode] = sum_temp

    # sum of off-diagonal part + sum of diagonal part
    Sum_DKinect[imode] = np.sum(DKinect[imode])

    if Calculate_Kernel == True:
        Sum_DKernel[imode] = np.sum(DKernel[imode])
        if calc_IBL_way == True:
            Sum_DKernel_IBL[imode] = np.sum(DKernel_IBL[imode])

report_ram()

# Convert from Ry/bohr to eV/A. Minus sign comes from F=-dV/du

Sum_DKinect_diag = -Sum_DKinect_diag*Ry2eV/bohr2A
Sum_DKinect = -Sum_DKinect*Ry2eV/bohr2A

if Calculate_Kernel == True:
    Sum_DKernel = -Sum_DKernel*Ry2eV/bohr2A
    if calc_IBL_way == True:
        Sum_DKernel_IBL = -Sum_DKernel_IBL*Ry2eV/bohr2A


# Warn if imag part is too big (>= 10^-6)

if max(abs(np.imag(Sum_DKinect_diag))) >= 1e-6:
    print('WARNING: Imaginary part of kinectic diagonal forces >= 10^-6 eV/angs!')

if max(abs(np.imag(Sum_DKinect))) >= 1e-6:
    print('WARNING: Imaginary part of kinectic forces >= 10^-6 eV/angs!')

if Calculate_Kernel == True:
    if max(abs(np.imag(Sum_DKernel))) >= 1e-6:
        print('WARNING: Imaginary part of Kernel forces >= 10^-6 eV/angs!')

    if calc_IBL_way == True:
        if max(abs(np.imag(Sum_DKernel_IBL))) >= 1e-6:
            print('WARNING: Imaginary part of Kernel (IBL) forces >= 10^-6 eV/angs!')

# Show just real part of numbers (default)


if show_imag_part == False:
    Sum_DKinect_diag = np.real(Sum_DKinect_diag)
    Sum_DKinect = np.real(Sum_DKinect)

    if Calculate_Kernel == True:
        Sum_DKernel = np.real(Sum_DKernel)
        if calc_IBL_way == True:
            Sum_DKernel_IBL = np.real(Sum_DKernel_IBL)

# Calculate forces cartesian basis

print("Calculating forces in cartesian basis")


if show_imag_part == True:
    F_cart_KE_IBL                       = np.zeros((Nat, 3), dtype=np.complex64)  # IBL just diag RPA
    F_cart_KE_David                     = np.zeros((Nat, 3), dtype=np.complex64)  # david thesis - diag + offdiag from kinect part
    if Calculate_Kernel == True:
        F_cart_Kernel_IBL                   = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's paper 
        F_cart_Kernel_IBL_correct           = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's with new kernel
else:
    F_cart_KE_IBL                       = np.zeros((Nat, 3))  # IBL just diag RPA
    F_cart_KE_David                     = np.zeros((Nat, 3))  # david thesis - diag + offdiag from kinect part
    if Calculate_Kernel == True:
        F_cart_Kernel_IBL                   = np.zeros((Nat, 3))  # Ismail-Beigi and Louie's paper 
        F_cart_Kernel_IBL_correct           = np.zeros((Nat, 3))  # Ismail-Beigi and Louie's with new kernel

for iatom in range(Nat):
    for imode in range(Nmodes):
        F_cart_KE_IBL[iatom] += Displacements[imode, iatom] * Sum_DKinect_diag[imode]
        F_cart_KE_David[iatom] += Displacements[imode, iatom] * Sum_DKinect[imode]

        if Calculate_Kernel == True:
            F_cart_Kernel_IBL[iatom] += Displacements[imode, iatom] * (Sum_DKernel_IBL[imode] + Sum_DKinect_diag[imode])
            F_cart_Kernel_IBL_correct[iatom] += Displacements[imode, iatom] * (Sum_DKernel[imode] + Sum_DKinect_diag[imode])


# Reporting forces in cartesian basis 
DIRECTION = ['x', 'y', 'z']

arq_out = open('forces_cart.out', 'w')

print('\n\nForces (eV/ang)\n')

if Calculate_Kernel == True:
    print('# Atom  dir  RPA_diag RPA_diag_offiag RPA_diag_Kernel RPA_diag_newKernel')
    arq_out.write('# Atom  dir  RPA_diag RPA_diag_offiag RPA_diag_Kernel RPA_diag_newKernel\n')
else:
    print('# Atom  dir  RPA_diag RPA_diag_offiag ')
    arq_out.write('# Atom  dir  RPA_diag RPA_diag_offiag \n')

for iatom in range(Nat):
    for idir in range(3):
        text =  str(iatom+1)+' '+DIRECTION[idir]+' '
        text += str(            F_cart_KE_IBL[iatom, idir])+' '
        text += str(          F_cart_KE_David[iatom, idir])+' '
        if Calculate_Kernel == True:
            text += str(        F_cart_Kernel_IBL[iatom, idir])+' '
            if calc_IBL_way == True:
                text += str(F_cart_Kernel_IBL_correct[iatom, idir])
        print(text)
        arq_out.write(text+'\n')

arq_out.close()

end_time = time.clock_gettime(0)

print('\n\nCalculation finished!')
print(f'Total time: '+report_time(start_time))

report_ram()

# stopping the library
tracemalloc.stop()