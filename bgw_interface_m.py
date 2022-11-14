
import h5py
import time
import numpy as np

from excited_forces_config import *


################ GW related functions #####################

def read_eqp_data(eqp_file, BSE_params):

    """Reads quasiparticle and dft energies results from sigma calculations.
    
    It is recommended to use the eqp.dat file from calculations using the
    absorption.flavor.x code, as it is compatible with the Acvk coefficients
    (same number of valence and conduction bands and same number of k points)

    Parameters:
    eqp_file (string)

    Returns:
        arrays : Eqp_val, Eqp_cond, Edft_val, Edft_cond
    """

    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds   = BSE_params.Ncbnds
    Nvbnds   = BSE_params.Nvbnds
    Nval     = BSE_params.Nval

    Eqp_val   = np.zeros((Nkpoints, Nvbnds), dtype=float)
    Edft_val  = np.zeros((Nkpoints, Nvbnds), dtype=float)
    Eqp_cond  = np.zeros((Nkpoints, Ncbnds), dtype=float)
    Edft_cond = np.zeros((Nkpoints, Ncbnds), dtype=float)

    print('Reading QP energies from eqp.dat file: ', eqp_file)
    arq = open(eqp_file)

    ik = -1

    for line in arq:
        linha = line.split()
        if linha[0] != '1':
            ik += 1
        else:
            iband_file = int(linha[1])

            if iband_file > Nval: # it is a cond band
                iband = iband_file - Nval 
                if iband <= Ncbnds:
                    #print('Cond -> iband_file iband Nval', iband_file, iband, Nval)
                    Edft_cond[ik, iband - 1] = float(linha[2])
                    Eqp_cond[ik, iband - 1] = float(linha[3])
            else: # it is val band
                iband = Nval - iband_file + 1
                if iband <= Nvbnds:
                    #print('Val -> iband_file iband Nval', iband_file, iband, Nval)
                    Edft_val[ik, iband - 1] = float(linha[2])
                    Eqp_val[ik, iband - 1] = float(linha[3])

    return Eqp_val, Eqp_cond, Edft_val, Edft_cond


################ BSE related functions ####################

def get_kernel(kernel_file, factor_head):

    """
    Reads the kernel matrix elements from BSE calculations
    Return the direct (Kd) and exchange (Kx) kernels in Ry
    """

    # start_time_func = time.clock_gettime(0)
    print('\nReading kernel matrix elements from ', kernel_file)

    # Kd = head (G=G'=0) + wing (G=0 or G'=0) + body (otherwise) - see https://doi.org/10.1016/j.cpc.2011.12.006

    f_hdf5 = h5py.File(kernel_file, 'r')

    celvol = f_hdf5['mf_header/crystal/celvol'][()]
    factor_kernel = -8.0*np.pi/celvol

    flavor_calc = f_hdf5['/bse_header/flavor'][()]

    Head = f_hdf5['mats/head'][()]
    Body = f_hdf5['mats/body'][()]
    Wing = f_hdf5['mats/wing'][()]
    Exchange = f_hdf5['mats/exchange'][()]

    if flavor_calc == 2:
        Kd =  (Head[:,:,:,:,:,:,0] + 1.0j*Head[:,:,:,:,:,:,1])*factor_head
        Kd += (Wing[:,:,:,:,:,:,0] + 1.0j*Wing[:,:,:,:,:,:,1])
        Kd += (Body[:,:,:,:,:,:,0] + 1.0j*Body[:,:,:,:,:,:,1])
        Kx =  -2*(Exchange[:,:,:,:,:,:,0] + 1.0j*Exchange[:,:,:,:,:,:,1])
    else:
        Kd =  (Head[:,:,:,:,:,:,0])*factor_head
        Kd += (Wing[:,:,:,:,:,:,0])
        Kd += (Body[:,:,:,:,:,:,0])
        Kx =  -2*(Exchange[:,:,:,:,:,:,0])        

    Kd = Kd*factor_kernel
    Kx = Kx*factor_kernel

    # end_time_func = time.clock_gettime(0)
    # print(f'Time spent on get_kernel function: '+report_time(start_time_func))

    return Kd, Kx



def get_exciton_info(exciton_file, iexc):

    """    
    Return the exciton energy and the eigenvec coefficients Acvk

    Assuming calculations with TD approximation
    Info about file at: http://manual.berkeleygw.org/3.0/eigenvectors_h5_spec/
    Also, just working for excitons with Q = 0 and one spin

    TODO -> for now calculting exciton info for exciton with index iexc
    but later, make it calculate for and set of exciton indexes
    
    Parameters:
    exciton_file = exciton file name (string). ex: eigenvecs.h5
    iexc = Exciton index to be read
    
    Returns:
    Acvk = Exciton wavefunc coefficients. array Akcv[ik, ic, iv] with complex values
    Omega = Exciton energy (BSE eigenvalue) in eV (float)
    """

    print('Reading exciton info from file', exciton_file)

    f_hdf5 = h5py.File(exciton_file, 'r')
    
    flavor_calc = f_hdf5['/exciton_header/flavor'][()]
    eigenvecs   = f_hdf5['exciton_data/eigenvectors'][()]          # (nQ, Nevecs, nk, nc, nv, ns, real or imag part)
    eigenvals   = f_hdf5['exciton_data/eigenvalues'][()] 

    Omega = eigenvals[iexc-1]
    if flavor_calc == 2:
        print('Flavor in BGW: complex')
        Akcv = eigenvecs[0,iexc-1,:,:,:,0,0] + 1.0j*eigenvecs[0,iexc-1,:,:,:,0,1]
    else:
        print('Flavor in BGW: real')
        Akcv = eigenvecs[0,iexc-1,:,:,:,0,0]

    print("    Max real value of Akcv: ", np.max(np.real(Akcv)))
    print("    Max imag value of Akcv: ", np.max(np.imag(Akcv)))
    print('\n\n')

    return Akcv, Omega


def get_params_from_eigenvecs_file(exciton_file):

    print('Reading parameters info from file', exciton_file)

    f_hdf5 = h5py.File(exciton_file, 'r')

    alat          = f_hdf5['/mf_header/crystal/alat'][()]            # lattice parameter in bohr
    cell_vol      = f_hdf5['/mf_header/crystal/celvol'][()]      # unit cell vol in bohr**3
    cell_vecs     = f_hdf5['/mf_header/crystal/avec'][()]       # lattice vectors (in units of alat)
    rec_cell_vecs = f_hdf5['/mf_header/crystal/bvec'][()]
    atomic_pos    = f_hdf5['/mf_header/crystal/apos'][()]      # in cartesian coordinates, in units of alat - important for visualization
    Nat           = f_hdf5['/mf_header/crystal/nat'][()]              # Number of atoms

    # Bands used to build BSE hamiltonian
    Nvbnds = f_hdf5['/exciton_header/params/nv'][()]  # Assuming TDA
    Ncbnds = f_hdf5['/exciton_header/params/nc'][()]

    # K points in the fine grid
    Kpoints_bse = f_hdf5['/exciton_header/kpoints/kpts'][()] 
    Nkpoints = f_hdf5['/exciton_header/kpoints/nk'][()]


################################################################################################
    ''''Getting Nval as norm of IFMAX
    In eigenvectors.h5 file, there is the IFMAX 
    the labels the highest occupied band in file. 
    For a semiconductor, IFMAX = Nval, independently of 
    the number of k points. I am assuming I am working with a semiconductor.
    In future I'll change it to be more general, but I need to change equations in theory
    to include occupations.'''

    ifmax_list = f_hdf5['/mf_header/kpoints/ifmax'][()][0] # ignoring spin degree of freedom!
    
    ifmax_values = []
    for ik in range(Nkpoints):
        if ifmax_values.count(ifmax_list[ik]) == 0:
            ifmax_values.append(ifmax_list[ik])
    
    Nval = min(ifmax_values)

    if len(ifmax_values) == 1:
        print(f' ---------> ifmax through k points is just one value ({ifmax_values[0]})')
    else:
        print('######################################################\n')
        print(f'WARNING! ifmax changes through k points! It means that the system is metallic, and we STILL did not implement it.')
        print('I will work with it as a semiconductor by setting the valence band to be min(ifmax) = {Nval}')
        print('######################################################\n')
################################################################################################

    # writing k points info to file - DEBUG

    if log_k_points == True:

        print('Writing k points in eigenvecs in Kpoints_eigenvecs_file')

        arq_kpoints = open('Kpoints_eigenvecs_file', 'w')

        for ik in range(len(Kpoints_bse)):
            kx, ky, kz = Kpoints_bse[ik]
            arq_kpoints.write(f'{kx}   {ky}   {kz}\n')

        arq_kpoints.close()

    # Reporting info from this file

    print(f'\nParameters from {exciton_file} :')

    print(f'    Total of atoms             = {Nat}')
    print(f'    Total of modes vib (3*Nat) = {3*Nat}')
    print(f'    Nkpoints                   = {Nkpoints}')
    print(f'    Number of cond bands       = {Ncbnds}')
    print(f'    Number of val bands        = {Nvbnds}')
    print(f'    Valence band index         = {Nval}')
    print('\n')
    print(f'    Lattice parameter (a.u.)   = {alat}')
    print(f'    Lattice vectors (in lattice parameter units): ')
    print(f'          a1 = ({cell_vecs[0, 0]}, {cell_vecs[0, 1]}, {cell_vecs[0, 2]})')
    print(f'          a2 = ({cell_vecs[1, 0]}, {cell_vecs[1, 1]}, {cell_vecs[1, 2]})')
    print(f'          a3 = ({cell_vecs[2, 0]}, {cell_vecs[2, 1]}, {cell_vecs[2, 2]})')
    print(f'    Reciprocal lattice vectors (2 * pi / lattice parameter):')
    print(f'          b1 = ({rec_cell_vecs[0, 0]}, {rec_cell_vecs[0, 1]}, {rec_cell_vecs[0, 2]})')
    print(f'          b2 = ({rec_cell_vecs[1, 0]}, {rec_cell_vecs[1, 1]}, {rec_cell_vecs[1, 2]})')
    print(f'          b3 = ({rec_cell_vecs[2, 0]}, {rec_cell_vecs[2, 1]}, {rec_cell_vecs[2, 2]})')
    print(f'\n\n')


    return Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_bse, Nkpoints, Nval, rec_cell_vecs


def get_params_from_alternative_file(alternative_params_file):

    file_params = open(alternative_params_file)
    rec_cell_vecs = []

    for line in file_params:

        linha = line.split()    

        if len(linha) > 1:
            if linha[0] == 'Nvbnds':
                Nvbnds = int(linha[1])
            elif linha[0] == 'Ncbnds':
                Ncbnds = int(linha[1])
            elif linha[0] ==  'Nval':
                Nval = int(linha[1])
            elif ['b1', 'b2', 'b3'].count(linha[0]) == 1:
                bx, by, bz = np.float(linha[1]), np.float(linha[2]), np.float(linha[3])
                rec_cell_vecs.append([bx, by, bz])
            elif linha[0] == 'Nat':
                Nat = int(linha[1])
            elif linha[0] == 'Nkpoints':
                Nkpoints = int(linha[1])
            elif linha[0] == 'alat': # alat in a.u.
                alat = float(linha[1])

    file_params.close()

    Kpoints_bse = []

    arq_kpoints_bse = open('kpoints_fine_bse')

    # 1   0.00000   0.00000   0.00000
    # 2   0.00000   0.00000   0.16667
    # 3  -0.00000  -0.00000   0.33333
    # 4  -0.00000   0.00000   0.50000
    # 5  -0.00000   0.00000  -0.33333


    for line in arq_kpoints_bse:
        linha = line.split()
        kx, ky, kz = float(linha[1]), float(linha[2]), float(linha[3])
        Kpoints_bse.append([kx, ky, kz])

    Kpoints_bse = np.array(Kpoints_bse)    
            
    atomic_pos = []
    cell_vol = 0.0
    cell_vecs = np.array([[0,0,0], [0,0,0], [0,0,0]])
    rec_cell_vecs = np.array(rec_cell_vecs)

    print('RECIPROCAL VECTORS')
    print(rec_cell_vecs)

    return Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_bse, Nkpoints, Nval, rec_cell_vecs


def get_exciton_info_alternative(Acvk_directory, iexc, Nkpoints, Ncbnds, Nvbnds):

    """
    When eigenvectors.h5 files are not available, must use this alternative here
    Have to use my modified version of summarize_eigenvectors code from BGW
    https://github.com/rafaeldgrande/utilities/blob/main/BGW/modified_summarize_eigenvectors.f90
    """

    exciton_file = Acvk_directory + f'/Avck_{iexc}'

    Akcv = np.zeros((Nkpoints, Ncbnds, Nvbnds), dtype=complex)

    print('Reading exciton info from file', exciton_file)
    arq = open(exciton_file)

    for line in arq:
        linha = line.split()
        if len(linha) == 6:
            if linha[0] != 'Special' and linha[0] != 'c':
                ic, iv, ik = int(linha[0]) - \
                    1, int(linha[1]) - 1, int(linha[2]) - 1
                if ic < Ncbnds and iv < Nvbnds:
                    Akcv[ik][ic][iv] = float(linha[3]) + 1.0j*float(linha[4])
            if linha[0] == 'Special':
                exc_energy = float(linha[-1])

    print('Exciton energy (eV): '+str(exc_energy)+'\n\n')

    return Akcv, exc_energy