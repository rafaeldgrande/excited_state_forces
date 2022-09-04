
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
print('*************************************************************\n\n')


# Importing modules
import numpy as np
import time
import tracemalloc # track ram usage
tracemalloc.start()


# excited state forces modules
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

# Classes 
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

# functions 

def get_BSE_MF_params():

    global MF_params, BSE_params, Nmodes
    global Nat, atomic_pos, cell_vecs, cell_vol, alat
    global Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval
    global rec_cell_vecs, Nmodes

    Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_BSE, Nkpoints_BSE, Nval, rec_cell_vecs = get_params_from_eigenvecs_file(exciton_file)
    Nmodes = 3 * Nat

    MF_params  = Parameters_MF(Nat, atomic_pos, cell_vecs, cell_vol, alat)
    BSE_params = Parameters_BSE(Nkpoints_BSE, Kpoints_BSE, Ncbnds, Nvbnds, Nval)

def report_expected_energies():

    Mean_Ekin = 0.0 
    if Calculate_Kernel == True:
        Mean_Kx, Mean_Kd = 0.0, 0.0

    for ik1 in range(BSE_params.Nkpoints_BSE):
        for ic1 in range(BSE_params.Ncbnds):
            for iv1 in range(BSE_params.Nvbnds):
                Mean_Ekin += (Eqp_cond[ik1, ic1] - Eqp_val[ik1, iv1])*abs(Akcv[ik1, ic1, iv1])**2

    if Calculate_Kernel == True:
        for ik1 in range(BSE_params.Nkpoints_BSE):
            for ic1 in range(BSE_params.Ncbnds):
                for iv1 in range(BSE_params.Nvbnds):
                    for ik2 in range(BSE_params.Nkpoints_BSE):
                        for ic2 in range(BSE_params.Ncbnds):
                            for iv2 in range(BSE_params.Nvbnds):
                                Mean_Kx += Ry2eV*np.conj(Akcv[ik1, ic1, iv1]) * Kx[ik2, ik1, ic2, ic1, iv2, iv1] * Akcv[ik2, ic2, iv2]
                                Mean_Kd += Ry2eV*np.conj(Akcv[ik1, ic1, iv1]) * Kd[ik2, ik1, ic2, ic1, iv2, iv1] * Akcv[ik2, ic2, iv2]

    print('Exciton energies (eV): ')
    print('    Omega         = ', Omega)
    print('    <KE>          = ', Mean_Ekin)
    print('    Omega - <KE>  = ', Omega - Mean_Ekin)
    if Calculate_Kernel == True:
        print('    <Kx>          = ', Mean_Kx)
        print('    <Kd>          = ', Mean_Kd)
        print('\n    DIFF          = ', Omega - (Mean_Ekin + Mean_Kd + Mean_Kx))


def correct_comp_vector(comp):
    # component is in alat units
    if comp < 0:
        return comp + 1
    elif comp > 1:
        return comp - 1
    else:
        return comp

def find_kpoint(kpoint, K_list):
    index_in_matrix = -1
    for index in range(len(K_list)):
        # if np.array_equal(kpoint, K_list[index]):
        if np.linalg.norm(kpoint - K_list[index]) <= TOL_DEG:
            index_in_matrix = index
    return index_in_matrix

def translate_bse_to_dfpt_k_points():
    ikBSE_to_ikDFPT = []
    # This list shows which k point from BSE corresponds to which 
    # point from DFPT calculation. 
    # ikBSE_to_ikDFPT[ikBSE] = ikDFPT
    # Means that the k point from eigenvectors.h5 with index ikBSE corresponds to 
    # the k point with index ikDFPT from DFPT calculation

    for ik in range(Nkpoints_BSE):

        # getting vectors from eigenvectors.h5 file in cartesian basis
        a1, a2, a3 = Kpoints_BSE[ik]

        # putting the vector in the first Brillouin zone
        a1 = correct_comp_vector(a1)
        a2 = correct_comp_vector(a2)
        a3 = correct_comp_vector(a3)

        # vector in cartesian basis
        vec_eigvecs = a1 * rec_cell_vecs[0] + a2 * rec_cell_vecs[1] + a3 * rec_cell_vecs[2]

        found_or_not = find_kpoint(vec_eigvecs, Kpoints_in_elph_file)
        # if found the vec_eigvecs in the Kpoints_in_elph_file, then returns 
        # the index in the Kpoints_in_elph_file. 
        # if did not find it, then returns -1

        # the conversion list from one to another
        ikBSE_to_ikDFPT.append(found_or_not)

    return ikBSE_to_ikDFPT

def check_k_points_BSE_DFPT():
# Checking if any kpoint is missing
    flag_missing_kpoints = False
    flag_repeated_kpoints = False

    if ikBSE_to_ikDFPT.count(-1) > 0:

        flag_missing_kpoints = True

        print('WARNING! Some k points from eigenvecs file were not found in the grid used in the DFPT calculation!')
        print(f'Total number of missing k points {ikBSE_to_ikDFPT.count(-1)}')
        print('The missing k points in DFPT are (in reciprocal lattice basis):')
        for ik in range(Nkpoints_BSE):
            if ikBSE_to_ikDFPT[ik] == -1:
                print(Kpoints_BSE[ik])
        

    # Checking if any kpoint is reported more than once

    for ikBSE in range(Nkpoints_BSE):
        how_many_times = 0  
        for ikBSE2 in range(Nkpoints_BSE):
            if np.linalg.norm(Kpoints_BSE[ikBSE] - Kpoints_BSE[ikBSE2]) <= TOL_DEG:
                how_many_times += 1
        if how_many_times > 1:
            print(f'WARNING!    This k point appear more than once: {Kpoints_BSE[ikBSE]} ')
            flag_repeated_kpoints = True

    if flag_missing_kpoints == False and flag_repeated_kpoints == False:
        print('Found no problem for k points from both DFPT and BSE calculations')
    else:
        print('Quiting program! Please check the above warnings!')
        quit()





########## RUNNING CODE ###################

start_time = time.clock_gettime(0)
# Getting BSE and MF parameters
# Reading eigenvecs.h5 file
get_BSE_MF_params()

# Getting exciton info
Akcv, Omega = get_exciton_info(exciton_file, iexc)

# getting info from eqp.dat (from absorption calculation)
Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, BSE_params)

# Getting kernel info from bsemat.h5 file
if Calculate_Kernel == True:
    Kd, Kx = get_kernel(kernel_file, factor_head)  

# Reporting expected energies
report_expected_energies()

# Getting elph coefficients
    
    # get displacement patterns
iq = 0 # FIXME -> generalize for set of q points
Displacements, Nirreps = get_patterns2(iq, MF_params)

    # get elph coefficients from .xml files
elph, Kpoints_in_elph_file = get_el_ph_coeffs(iq, Nirreps)

    # apply acoustic sum rule
elph = impose_ASR(elph, Displacements, MF_params, acoutic_sum_rule)

    # filter data to get just g_c1c2 and g_v1v2
elph_cond, elph_val = filter_elph_coeffs(elph, MF_params, BSE_params)

# report_ram()
# print('DELETING ELPH')
del elph
report_ram()


# Checking kpoints from DFPT and BSE calculations
# The kpoints in eigenvecs.h5 are not in the same order in the 
# input for the fine grid calculation. 
# The k points in BSE are reported in reciprocal lattice vectors basis
# and in DFPT those k points are reported in cartersian basis in units 
# of reciprocal lattice

# It SEEMS that the order of k points in the eqp.dat (produced by the absorption code)
# is the same than the order of k points in the eigenvecs file
# Maybe it would be necessary to check it later!

# First let's put all k points from BSE grid in the first Brillouin zone

print('Checking if kpoints of DFPT and BSE agree with each other')
ikBSE_to_ikDFPT = translate_bse_to_dfpt_k_points()


# Now checking if everything is ok with ikBSE_to_ikDFPT list
# if something is wrong kill the code
check_k_points_BSE_DFPT()

report_ram()

########## Calculating stuff ############

print("Calculating matrix elements for forces calculations <cvk|dH/dx_mu|c'v'k'>")

# Creating auxialiry quantities
# aux_cond_matrix[imode, ik, ic1, ic2] = elph_cond[imode, ik, ic1, ic2] * deltaEqp / deltaEdft (if ic1 != ic2)
# aux_val_matrix[imode, ik, iv1, iv2]  = elph_val[imode, ik, iv1, iv2]  * deltaEqp / deltaEdft (if iv1 != iv2)
# If ic1 == ic2 (iv1 == iv2), then the matrix elements are just the elph coefficients"""
aux_cond_matrix, aux_val_matrix = aux_matrix_elem(elph_cond, elph_val, Eqp_val, Eqp_cond, Edft_val, Edft_cond, MF_params, BSE_params, ikBSE_to_ikDFPT)

# Calculating matrix elements F_cvkc'v'k'
DKinect = calc_Dkinect_matrix(Akcv, aux_cond_matrix, aux_val_matrix, MF_params, BSE_params)

del aux_cond_matrix, aux_val_matrix

# Kernel derivatives
if Calculate_Kernel == True:

    EDFT = Edft_val, Edft_cond
    EQP = Eqp_val, Eqp_cond
    ELPH = elph_cond, elph_val

    if calc_IBL_way == True:
        DKernel, DKernel_IBL = calc_deriv_Kernel((Kx+Kd)*Ry2eV, EDFT, EQP, ELPH, Akcv, MF_params, BSE_params)
    else:
        DKernel = calc_deriv_Kernel((Kx+Kd)*Ry2eV, EDFT, EQP, ELPH, Akcv, MF_params, BSE_params)

    del Kx, Kd

print("Calculating sums")

Sum_DKinect_diag            = np.zeros((Nmodes), dtype=complex)
Sum_DKinect                 = np.zeros((Nmodes), dtype=complex)

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
    Sum_DKernel     = np.zeros((Nmodes), dtype=np.complex64)
    Sum_DKernel_IBL = np.zeros((Nmodes), dtype=np.complex64)

    for imode in range(Nmodes):
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
    F_cart_KE_IBL                       = np.zeros((Nat, 3), dtype=complex)  # IBL just diag RPA
    F_cart_KE_David                     = np.zeros((Nat, 3), dtype=complex)  # david thesis - diag + offdiag from kinect part
    if Calculate_Kernel == True:
        F_cart_Kernel_IBL                   = np.zeros((Nat, 3), dtype=complex)  # Ismail-Beigi and Louie's paper 
        F_cart_Kernel_IBL_correct           = np.zeros((Nat, 3), dtype=complex)  # Ismail-Beigi and Louie's with new kernel
else:
    F_cart_KE_IBL                       = np.zeros((Nat, 3))  # IBL just diag RPA
    F_cart_KE_David                     = np.zeros((Nat, 3))  # david thesis - diag + offdiag from kinect part
    if Calculate_Kernel == True:
        F_cart_Kernel_IBL                   = np.zeros((Nat, 3))  # Ismail-Beigi and Louie's paper 
        F_cart_Kernel_IBL_correct           = np.zeros((Nat, 3))  # Ismail-Beigi and Louie's with new kernel

for iatom in range(Nat):
    for imode in range(Nmodes):
        F_cart_KE_IBL[iatom]   += Displacements[imode, iatom] * Sum_DKinect_diag[imode]
        F_cart_KE_David[iatom] += Displacements[imode, iatom] * Sum_DKinect[imode]

        if Calculate_Kernel == True:
            F_cart_Kernel_IBL[iatom]         = F_cart_Kernel_IBL[iatom]         + Displacements[imode, iatom] * (Sum_DKernel_IBL[imode] + Sum_DKinect_diag[imode])
            F_cart_Kernel_IBL_correct[iatom] = F_cart_Kernel_IBL_correct[iatom] + Displacements[imode, iatom] * (Sum_DKernel[imode]     + Sum_DKinect_diag[imode])
            # need to make x = x + y, instead of x += y because numpy complains that x+=y does not work when type(x)=!type(y) (one is complex and the other is real - float)


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
report_ram()
# stopping the library
tracemalloc.stop()

print('\n\nCalculation finished!')
print(f'Total time: '+report_time(start_time))