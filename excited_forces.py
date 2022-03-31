
PLOT = False

# TO DO
# 1 - incluir derivadas do kernel. ler arquivos hdf5
# 2 - checar se termos nao diagonais incluidos aqui estao certos
# 3 - criar modulo pra esse codigo aqui
# 4 - ler dados dos calculos dfpt usando modulo pra ler arquivos .xml
# 5 - ler dados de excitons lendo arquivos binarios

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


print('---- Parameters -----')
print('Number of atoms : ' + str(Nat))
print('Number of modes (3*Nat) : ', Nmodes)
print('Nvbnds = '+str(Nvbnds) + ', Ncbnds = '+str(Ncbnds))
print('Valence band : ', Nval)
print('Nkpts = '+str(Nkpoints))
print('Exciton index to be read : '+str(iexc))
if calc_IBL_way == True:
    print('Calculating derivatives of Kernel using Ismail-Beigi and Louie\'s paper approach')
print('---------------------')

# Variables 

Shape = (Nmodes, Nkpoints, Ncbnds, Nvbnds)
Shape2 = (Nmodes, Nkpoints, Ncbnds, Nvbnds, Nkpoints, Ncbnds, Nvbnds)

DKinect_diag     = np.zeros(Shape, dtype=np.complex64)
DKinect_offdiag  = np.zeros(Shape, dtype=np.complex64)
DKernel          = np.zeros(Shape2, dtype=np.complex64)
DKernel_IBL      = np.zeros(Shape2, dtype=np.complex64)

Sum_DKinect_diag       = np.zeros((Nmodes), dtype=np.complex64)
Sum_DKinect_offdiag    = np.zeros((Nmodes), dtype=np.complex64)
Sum_DKernel            = np.zeros((Nmodes), dtype=np.complex64)
Sum_DKernel_IBL        = np.zeros((Nmodes), dtype=np.complex64)

Forces_disp           = np.zeros((Nmodes), dtype=np.complex64)

Forces_modes          = np.zeros((Nmodes), dtype=np.complex64)

############ Getting info from files #############

# getting info from eqp.dat

Eqp_val, Eqp_cond, Edft_val, Edft_cond = read_eqp_data(eqp_file, Nkpoints, Nvbnds, Ncbnds, Nval)

# Getting exciton info

# Akcv = get_exciton_info(exciton_file, Nkpoints, Nvbnds, Ncbnds)
Akcv, exc_energy = get_exciton_info(exciton_file, Nkpoints, Nvbnds, Ncbnds)
print('Akcv', np.shape(Akcv))

# get displacement patterns

iq = 0 # FIXME -> generalize for set of q points
Displacements, Nirreps, Perts = get_patterns(el_ph_dir, iq, Nmodes, Nat)
elph_val, elph_cond, elph_aux = get_el_ph_coeffs(el_ph_dir, iq, Nirreps, Perts, Nmodes, Nkpoints, Ncbnds, Nvbnds, Nval)

# Get kernel info

Kx, Kd = get_kernel(kernel_file)

########## Calculating stuff ############

# Creating auxialiry quantities

# Derivatives of diagonal kinect part

aux_diag = np.zeros(Shape, dtype=np.complex64)  # <ck|dV/du_mode|ck> - <vk|dV/du_mode|vk>

for imode in range(Nmodes):
    for ik in range(Nkpoints):
        for ic in range(Ncbnds):
            cond_term = elph_cond[imode][ik][ic][ic]
            for iv in range(Nvbnds):
                val_term = elph_val[imode][ik][iv][iv]
                aux_diag[imode][ik][ic][iv] = cond_term - val_term

# Derivatives of offdiagonal kinect part
# aux_offdiag_cond_cvk = sum_{c' != c} <A|c'vk><c'k|dV/du_mode| ck> (Eqp_ck - Eqp_c'k) / (Edft_ck - Edft_c'k)
# aux_offdiag_val_cvk  = sum_{v' != v} <A|cv'k><vk |dV/du_mode|v'k> (Eqp_v'k - Eqp_vk) / (Edft_v'k - Edft_vk)

aux_offdiag_cond = np.zeros((Nmodes, Nkpoints, Ncbnds, Nvbnds), dtype=np.complex64)
aux_offdiag_val  = np.zeros((Nmodes, Nkpoints, Ncbnds, Nvbnds), dtype=np.complex64)

for imode in range(Nmodes):
    for ik in range(Nkpoints):
        for ic in range(Ncbnds):
            for iv in range(Nvbnds):
                offdiag_cond_temp = 0
                for icp in range(Ncbnds):
                    if abs(Edft_cond[ik][ic] - Edft_cond[ik][icp]) > TOL_DEG:
                        temp = np.conj(Akcv[ik][icp][iv])*elph_cond[imode][ik][icp][ic]
                        temp = temp*(Eqp_cond[ik][ic] - Eqp_cond[ik][icp]) / (Edft_cond[ik][ic] - Edft_cond[ik][icp])
                        offdiag_cond_temp += temp
                        #print('cont, imode, ik, ic, icp', cont, imode, ik, ic, icp)
                aux_offdiag_cond[imode][ik][ic][iv] = offdiag_cond_temp

                offdiag_val_temp = 0
                for ivp in range(Nvbnds):
                    if abs(Edft_val[ik][ivp] - Edft_val[ik][iv]) > TOL_DEG:
                        temp = np.conj(Akcv[ik][ic][ivp])*elph_val[imode][ik][iv][ivp]
                        temp = temp*(Eqp_val[ik][ivp] - Eqp_val[ik][iv]) / (Edft_val[ik][ivp] - Edft_val[ik][iv])
                        offdiag_val_temp += temp
                        #print('cont, imode, ik, iv, ivp', cont, imode, ik, iv, ivp)
                aux_offdiag_val[imode][ik][ic][iv] = offdiag_val_temp 

# Compute diag elements - kinect part
for imode in range(Nmodes):
    DKinect_diag[imode] = aux_diag[imode]*abs(Akcv)**2

# Compute offdiag elements - kinect part 
for imode in range(Nmodes):
    DKinect_offdiag[imode] = (aux_offdiag_cond[imode] - aux_offdiag_val[imode])*Akcv


# Forces from Kernel derivatives

EDFT = Edft_val, Edft_cond
EQP = Eqp_val, Eqp_cond
Nparams = Ncbnds, Nvbnds, Nkpoints
ELPH = elph_cond, elph_val

if write_DKernel == True:
    arq_Dkernel = open('Dkernel', 'w')
    if calc_IBL_way == True:
        arq_Dkernel.write('ik1, ik2, iv1, iv2, ic1, ic2, imode, DK_new, DK_old\n')
    else:
        arq_Dkernel.write('ik1, ik2, iv1, iv2, ic1, ic2, imode, DK_new\n')

for imode in range(Nmodes):
    for ik1 in range(Nkpoints):
        for ic1 in range(Ncbnds):
            for iv1 in range(Nvbnds):

                A_bra = np.conj(Akcv[ik1][ic1][iv1])

                for ik2 in range(Nkpoints):
                    for ic2 in range(Ncbnds):
                        for iv2 in range(Nvbnds):

                            A_ket = Akcv[ik2][ic2][iv2]

                            indexes = ik1, ik2, iv1, iv2, ic1, ic2, imode
                            dK = calc_DKernel(indexes, Kx + Kd, calc_IBL_way, EDFT, EQP, ELPH, Nparams, TOL_DEG)

                            if calc_IBL_way == False:
                                DKernel[imode][ik1][ic1][iv1][ik2][ic2][iv2] = A_bra * dK * A_ket
                            else:
                                DKernel[imode][ik1][ic1][iv1][ik2][ic2][iv2] = A_bra * dK[0] * A_ket
                                DKernel_IBL[imode][ik1][ic1][iv1][ik2][ic2][iv2] = A_bra * dK[1] * A_ket

                            if write_DKernel == True:
                                for item in indexes:
                                    arq_Dkernel.write(str(item)+' ')
                                if calc_IBL_way == True:
                                    arq_Dkernel.write(str(dK[0])+' '+str(dK[1])+'\n')
                                else:
                                    arq_Dkernel.write(str(dK)+'\n')

if write_DKernel == True:
    arq_Dkernel.close()

# Sums
for imode in range(Nmodes):
    Sum_DKinect_diag[imode] = np.sum(DKinect_diag[imode])
    Sum_DKinect_offdiag[imode] = np.sum(DKinect_offdiag[imode])
    Sum_DKernel[imode] = np.sum(DKernel[imode])
    if calc_IBL_way == True:
        Sum_DKernel_IBL[imode] = np.sum(DKernel_IBL[imode])


# Convert to eV/A. Minus sign comes from F=-dV/du
Sum_DKinect_diag = -Sum_DKinect_diag*Ry2eV/bohr2A
Sum_DKinect_offdiag = -Sum_DKinect_offdiag*Ry2eV/bohr2A
Sum_DKernel = -Sum_DKernel*Ry2eV/bohr2A
if calc_IBL_way == True:
    Sum_DKernel_IBL = -Sum_DKernel_IBL*Ry2eV/bohr2A

# Calculate forces cartesian basis

F_cart_KE_IBL                       = np.zeros((Nat, 3), dtype=np.complex64)  # david thesis - diag + offdiag from kinect part
F_cart_KE_David                     = np.zeros((Nat, 3), dtype=np.complex64)  # david thesis - diag + offdiag from kinect part + derivative of Kernel (corrected)
F_cart_Kernel_IBL                   = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's paper 
F_cart_Kernel_IBL_correct           = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's paper with dK = 0
F_cart_Kernel_IBL_correct_extended  = np.zeros((Nat, 3), dtype=np.complex64)  # Ismail-Beigi and Louie's paper with new correct dK 

for iatom in range(Nat):
    for imode in range(Nmodes):
        F_cart_KE_IBL[iatom] += Displacements[imode][iatom]* Sum_DKinect_diag[imode]
        F_cart_KE_David[iatom] += Displacements[imode][iatom]*( Sum_DKinect_diag[imode] + Sum_DKinect_offdiag[imode] )
        F_cart_Kernel_IBL[iatom] += Displacements[imode][iatom]*( Sum_DKernel_IBL[imode] )
        F_cart_Kernel_IBL_correct[iatom] += Displacements[imode][iatom]*( Sum_DKernel[imode])

#print('\n\n\n################# Forces (eV/A) in cartesian basis #####################')

DIRECTION = ['x', 'y', 'z']

arq_out = open('forces_cart.out', 'w')


arq_out.write('# Atom  dir  KE_IBL KE_David Kernel_IBL Kernel_IBL_correct\n')

for iatom in range(Nat):
    for idir in range(3):
        arq_out.write(str(iatom+1)+' '+DIRECTION[idir]+' '+
                      str(F_cart_KE_IBL[iatom][idir])+' '+
                      str(F_cart_KE_David[iatom][idir])+' '+
                      str(F_cart_Kernel_IBL[iatom][idir])+' '+
                      str(F_cart_Kernel_IBL_correct[iatom][idir])+'\n')

arq_out.close()


# print('\n David thesis\' approach\n')
# for iatom in range(Nat):
#     print('Atom '+str(iatom+1)+' '+
#           str(Forces_cartesian_DT[iatom][0])+'  '+str(Forces_cartesian_DT[iatom][1])+'  '+str(Forces_cartesian_DT[iatom][2]))

# print('\n David thesis\' with Kernel approach\n')
# for iatom in range(Nat):
#     print('Atom '+str(iatom+1)+' '+
#           str(Forces_cartesian_DT_dK[iatom][0])+'  '+str(Forces_cartesian_DT_dK[iatom][1])+'  '+str(Forces_cartesian_DT_dK[iatom][2]))

# print('\n Ismail-Beigi Louie (wo Kernel) approach\n')
# for iatom in range(Nat):
#     print('Atom '+str(iatom+1)+' '+
#           str(Forces_cartesian_IBL_nodK[iatom][0])+'  '+str(Forces_cartesian_IBL_nodK[iatom][1])+'  '+str(Forces_cartesian_IBL_nodK[iatom][2]))

# if calc_IBL_way == True:
#     print('\n Ismail-Beigi Louie (w old Kernel) approach\n')
#     for iatom in range(Nat):
#         print('Atom '+str(iatom+1)+' '+
#             str(Forces_cartesian_IBL_dKold[iatom][0])+'  '+str(Forces_cartesian_IBL_dKold[iatom][1])+'  '+str(Forces_cartesian_IBL_dKold[iatom][2]))

# print('\n Ismail-Beigi Louie (w new correct Kernel derivative) approach\n')
# for iatom in range(Nat):
#     print('Atom '+str(iatom+1)+' '+
#           str(Forces_cartesian_IBL_dKnew[iatom][0])+'  '+str(Forces_cartesian_IBL_dKnew[iatom][1])+'  '+str(Forces_cartesian_IBL_dKnew[iatom][2]))

# # Calculate forces displacement basis

# for imode in range(Nmodes):
#     Forces_disp[imode] =  Sum_DKinect_diag[imode] + Sum_DKinect_offdiag[imode]

# # Calculate forces in modes basis 

# if calc_modes_basis is True:

#     # Matrix to convert from displacement to cartesian

#     disp2cart = np.zeros((Nmodes, Nmodes), dtype=np.complex64)
#     for imode in range(Nmodes):
#         imodep = 0
#         for iat in range(Nat):
#             for idir in range(3):
#                 disp2cart[imodep][imode] = Displacements[imode][iat][idir]
#                 imodep += 1
#     #print('disp2cart \n', disp2cart)

#     # Read eigenvecs - FIXME -> generalize for several q's
#     modes2cart = get_modes2cart_matrix(dyn_file, Nat, Nmodes)
#     cart2modes = np.linalg.inv(modes2cart)
#     #print('cart2modes \n', modes2cart)
#     #print('determinantes :', np.linalg.det(cart2modes), np.linalg.det(modes2cart))


#     #print(np.real(modes2cart))
#     #print(np.real(disp2cart))

#     disp2modes = cart2modes @ disp2cart
#     Forces_modes = disp2modes @ Forces_disp
#     Forces_modes_IL = disp2modes @ Forces_disp_IL

#     print('Disp2modes matrix')
#     np.set_printoptions(precision=2)
#     print(disp2modes)

# # convert to real numbers if imag part is too small - flag controlled
# if just_real is True:
#     DKinect_diag         = np.real(DKinect_diag)
#     DKinect_offdiag      = np.real(DKinect_offdiag)
#     Displacements       = np.real(Displacements)
#     Forces_cartesian    = np.real(Forces_cartesian)
#     Forces_cartesian_IL = np.real(Forces_cartesian_IL)
#     Forces_disp         = np.real(Forces_disp)
#     Forces_disp_IL      = np.real(Forces_disp_IL)
#     if calc_modes_basis is True:
#         Forces_modes        = np.real(Forces_modes)
#         Forces_modes_IL     = np.real(Forces_modes_IL)


# # print forces in displacements basis
# print('\n\n\n################# Forces (eV/A) in displacement basis #####################')

# print('\n David thesis approach')
# for imode in range(Nmodes):
#     print('Displacement '+str(imode+1)+' : ', Sum_DKinect_diag[imode] + Sum_DKinect_offdiag[imode])

# print('\n\n\n################# Forces (eV/A) in cartesian basis #####################')

# print('\n David thesis approach\n')
# for iatom in range(Nat):
#     print('Atom '+str(iatom+1)+' '+
#           str(Forces_cartesian[iatom][0])+'  '+str(Forces_cartesian[iatom][1])+'  '+str(Forces_cartesian[iatom][2]))

# print('\n\n Ismail Louie approach \n')
# for iatom in range(Nat):
#     print('Atom '+str(iatom+1)+' '+
#           str(Forces_cartesian_IL[iatom][0])+'  '+str(Forces_cartesian_IL[iatom][1])+'  '+str(Forces_cartesian_IL[iatom][2]))


# if calc_modes_basis is True:
#     print('\n\n\n################# Forces (eV/A) in modes basis #####################')

#     print('\n David thesis approach\n')
#     for imode in range(Nmodes):
#         print('Mode ', imode + 1, Forces_modes[imode])

#     print('\n\n Ismail Louie approach \n')
#     for imode in range(Nmodes):
#         print('Mode ', imode + 1, Forces_modes_IL[imode])


if PLOT is True:
    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rc('lines', linewidth=3.0, markersize=10.0)
    plt.rc('text', usetex=True)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIG_SIZE = 18
    fator_fig = 2.2

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=BIG_SIZE)
    plt.rc('figure', figsize=(8, 6))
    plt.rc('figure', autolayout=True)

    PDF_NAMES = ['DKinect_diag_exc'+str(iexc)+'.pdf', 'DKinect_offdiag_exc'+str(iexc)+'.pdf', 'DKinect_offdiagIL_exc'+str(iexc)+'.pdf',
                 'elph_cond.pdf', 'elph_val.pdf']
    Forces = [DKinect_diag, DKinect_offdiag, elph_cond, elph_val]
    FirstTitle = [r'$F(\mathrm{eV/\AA})$', r'$F(\mathrm{eV/\AA})$', r'$F(\mathrm{eV/\AA})$', 
                  r"$\langle ck | dV_{\mathrm{scf}}/du | c'k \rangle (\mathrm{Ry/bohr})$", 
                  r"$\langle vk | dV_{\mathrm{scf}}/du | v'k \rangle (\mathrm{Ry/bohr})$"]

    VTICKS, CTICKS = [], []
    for i in range(Nvbnds):
        VTICKS.append(str(i+1))
    for i in range(Ncbnds):
        CTICKS.append(str(i+1))

    maxVal = np.amax(abs(Akcv))
    for ik in range(Nkpoints):
        fig, axs = plt.subplots(1, 3, dpi=200)
        fig.suptitle(r'$A_{kcv}$'+'- k point ' + str(ik + 1) + '- '+r'$E_{ex} =$' + str(exc_energy)+' eV', fontsize=18)

        plt.sca(axs[0])
        plt.title('Re', fontsize=16)
        im = plt.imshow(np.real(Akcv[ik]), cmap="seismic", vmin=-maxVal, vmax=maxVal)
        #cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        #cb.formatter.set_powerlimits((0, 0))
        #cb.update_ticks()
        add_colorbar(im)
        plt.xticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)
        plt.yticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
        plt.xlabel('Val')
        plt.ylabel('Cond')

        plt.sca(axs[1])
        plt.title('Imag', fontsize=16)
        im = plt.imshow(np.imag(Akcv[ik]), cmap="seismic", vmin=-maxVal, vmax=maxVal)
        #cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        #cb.formatter.set_powerlimits((0, 0))
        #cb.update_ticks()
        add_colorbar(im)
        plt.xticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)
        plt.yticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
        plt.xlabel('Val')

        plt.sca(axs[2])
        plt.title('Abs', fontsize=16)
        im = plt.imshow(abs(Akcv[ik]), cmap="Greys", vmin=0, vmax=maxVal)
        #cb = plt.colorbar(im, fraction=0.046, pad=0.04)
        #cb.formatter.set_powerlimits((0, 0))
        #cb.update_ticks()
        add_colorbar(im)
        plt.xticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)
        plt.yticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
        plt.xlabel('Val')   

        plt.savefig('Acvk_'+str(iexc)+'.png')

    for i in range(5):
        with PdfPages(PDF_NAMES[i]) as pdf:

            if i <= 2:
                maxVal = np.amax(abs(Forces[i]))
            else:
                maxVal = max( np.amax(abs(elph_val)), np.amax(abs(elph_cond)) )
            #force_name = force_fig_title[i]

            for imode in range(Nmodes):
                for ik in range(Nkpoints):

                    data_plot = Forces[i][imode][ik]

                    #fig, axs = plt.subplots(1, 3, dpi=200, figsize=(3*Nvbnds, Ncbnds + 4))
                    fig, axs = plt.subplots(1, 3, dpi=200)
                    fig.suptitle(FirstTitle[i]+' - mode ' + str(imode + 1) + ', k point ' + str(ik + 1), fontsize=18)

                    plt.sca(axs[0])
                    plt.title('Re', fontsize=16)
                    im = plt.imshow(np.real(data_plot), cmap="seismic", vmin=-maxVal, vmax=maxVal)
                    #cb = plt.colorbar(im, fraction=0.046, pad=0.04)
                    #cb.formatter.set_powerlimits((0, 0))
                    #cb.update_ticks()
                    add_colorbar(im)
                    if i <= 2:
                        plt.xticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)
                        plt.yticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
                        plt.ylabel('Cond')
                        plt.xlabel('Val')
                    if i == 3:
                        plt.xlabel('Cond')
                        plt.yticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
                        plt.xticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
                    if i == 4:
                        plt.xlabel('Val')
                        plt.yticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)
                        plt.xticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)

                    plt.sca(axs[1])
                    plt.title('Imag', fontsize=16)
                    im = plt.imshow(np.imag(data_plot), cmap="seismic", vmin=-maxVal, vmax=maxVal)
                    #cb = plt.colorbar(im, fraction=0.046, pad=0.04)
                    #cb.formatter.set_powerlimits((0, 0))
                    #cb.update_ticks()
                    add_colorbar(im)
                    if i <= 2:
                        plt.xticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)
                        plt.yticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
                        plt.xlabel('Val')
                    if i == 3:
                        plt.xlabel('Cond')
                        plt.yticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
                        plt.xticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
                    if i == 4:
                        plt.xlabel('Val')
                        plt.yticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)
                        plt.xticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)

                    plt.sca(axs[2])
                    plt.title('Abs', fontsize=16)
                    im = plt.imshow(abs(data_plot), cmap="Greys", vmin=0, vmax=maxVal)
                    #cb = plt.colorbar(im, fraction=0.046, pad=0.04)
                    #cb.formatter.set_powerlimits((0, 0))
                    #cb.update_ticks()
                    add_colorbar(im)
                    if i <= 2:
                        plt.xticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)
                        plt.yticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
                        plt.xlabel('Val')
                    if i == 3:
                        plt.xlabel('Cond')
                        plt.yticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
                        plt.xticks(np.linspace(0, Ncbnds-1, num=Ncbnds), CTICKS)
                    if i == 4:
                        plt.xlabel('Val')
                        plt.yticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)
                        plt.xticks(np.linspace(0, Nvbnds-1, num=Nvbnds), VTICKS)                        
                    pdf.savefig()
                    plt.close()


print('\n\nFinished!')
