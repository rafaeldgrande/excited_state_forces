
from excited_forces_config import *
from modules_to_import import *


################ GW related functions ############################

def read_eqp_data(eqp_file, BSE_params):
    """
    Read quasiparticle and DFT energies from sigma calculations from file eqp.dat. Needs to be 
    the results on the fine grid. This file is produced by the absorption code, where it 
    interpolates eqp_co.dat to a fine grid. 

    Parameters:
    eqp_file (str): Path to the eqp.dat file.
    BSE_params (namedtuple): Named tuple containing the following fields:
        - Nkpoints_BSE (int): Number of k-points used in the BSE calculations.
        - Ncbnds (int): Number of conduction bands used in BSE calculations.
        - Nvbnds (int): Number of valence bands used in BSE calculation.
        - Nval (int): Total number of valence electrons.

    Returns:
    Tuple containing the following arrays:
        - Eqp_val (numpy array): Quasiparticle energies of valence bands for each k-point.
        - Eqp_cond (numpy array): Quasiparticle energies of conduction bands for each k-point.
        - Edft_val (numpy array): DFT energies of valence bands for each k-point.
        - Edft_cond (numpy array): DFT energies of conduction bands for each k-point.
    """
    
    # Unpack the BSE parameters
    Nkpoints = BSE_params.Nkpoints_BSE
    Ncbnds = BSE_params.Ncbnds
    Nvbnds = BSE_params.Nvbnds
    Nval = BSE_params.Nval
    
    # Initialize the arrays to store the energies
    Eqp_val = np.zeros((Nkpoints, Nvbnds), dtype=float)
    Eqp_cond = np.zeros((Nkpoints, Ncbnds), dtype=float)
    Edft_val = np.zeros((Nkpoints, Nvbnds), dtype=float)
    Edft_cond = np.zeros((Nkpoints, Ncbnds), dtype=float)
    
    print(f'Reading QP energies from {eqp_file}')
    
    with open(eqp_file, 'r') as f:
        # Start with k-point index -1, since the first line of the file
        # does not contain energies
        ik = -1
        
        for line in f:
            linha = line.split()
            # If the first entry in the line is not 1, we have a new k-point
            if linha[0] != '1':
                ik += 1
            else:
                iband_file = int(linha[1])

                if iband_file > Nval:
                    # This is a conduction band
                    iband = iband_file - Nval 
                    if iband <= Ncbnds:
                        # Store the DFT and QP energies for the conduction band
                        Edft_cond[ik, iband-1] = float(linha[2])
                        Eqp_cond[ik, iband-1] = float(linha[3])
                else:
                    # This is a valence band
                    iband = Nval - iband_file + 1
                    if iband <= Nvbnds:
                        # Store the DFT and QP energies for the valence band
                        Edft_val[ik, iband-1] = float(linha[2])
                        Eqp_val[ik, iband-1] = float(linha[3])
    
    return Eqp_val, Eqp_cond, Edft_val, Edft_cond


################ BSE related functions ####################

def get_kernel(kernel_file, factor_head):

    """
    Reads the kernel matrix elements from BSE calculations and returns the
    direct (Kd) and exchange (Kx) kernels in Ry.

    Parameters:
    kernel_file (str): path to the kernel file
    factor_head (float): factor to be applied to the head part of the kernel
    spin_triplet (bool): whether the calculation includes spin triplet - K = Kd
    local_fields (bool): in this case the kernel is just K = Kx

    Returns:
    Kd (ndarray): direct kernel matrix elements in Ry
    Kx (ndarray): exchange kernel matrix elements in Ry
    """
    print(f'Reading kernel matrix elements from {kernel_file}')

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

    if local_fields == True:
        Kd = Kd * 0
    else:
        Kd = Kd*factor_kernel

    if spin_triplet == True:
        Kx = Kx * 0.0
    else:
        Kx = Kx*factor_kernel

    # end_time_func = time.clock_gettime(0)
    # print(f'Time spent on get_kernel function: '+report_time(start_time_func))

    return Kd, Kx

def get_params_Kernel(kernel_file):
    
    """
    Reads parameters for BSE calculation from Kernel file (bsemat.h5)
    """
    
    f_hdf5 = h5py.File(kernel_file, 'r')
    
    Nkpoints_BSE = f_hdf5['/bse_header/kpoints/nk'][()]
    Kpoints_BSE = f_hdf5['/bse_header/kpoints/kpts'][()]
    Nvbnds = f_hdf5['/bse_header/bands/nvb'][()]
    Ncbnds = f_hdf5['/bse_header/bands/ncb'][()]
    
    return Nkpoints_BSE, Kpoints_BSE, Nvbnds, Ncbnds
    
    

def get_exciton_info(exciton_file, iexc):

    """    
    Return the exciton energy and the eigenvec coefficients Acvk

    Assuming calculations with TD approximation
    Info about file at: http://manual.berkeleygw.org/3.0/eigenvectors_h5_spec/
    Also, just working for excitons with Q = 0 and one spin
    
    Parameters:
    exciton_file = exciton file name (string). ex: eigenvecs.h5
    iexc = Exciton index to be read
    
    Returns:
    Acvk = Exciton wavefunc coefficients. array Akcv[ik, ic, iv] with complex values
    Omega = Exciton energy (BSE eigenvalue) in eV (float)
    """
    f_hdf5 = h5py.File(exciton_file, 'r')
    
    flavor_calc = f_hdf5['/exciton_header/flavor'][()]
    eigenvecs   = f_hdf5['exciton_data/eigenvectors'][()]          # (nQ, Nevecs, nk, nc, nv, ns, real or imag part)

    if flavor_calc == 2:
        Akcv = eigenvecs[0,iexc-1,:,:,:,0,0] + 1.0j*eigenvecs[0,iexc-1,:,:,:,0,1]
    else:
        Akcv = eigenvecs[0,iexc-1,:,:,:,0,0]

    return Akcv


def get_params_from_eigenvecs_file(exciton_file):

    print('Reading parameters info from file', exciton_file)

    f_hdf5 = h5py.File(exciton_file, 'r')

    alat          = f_hdf5['/mf_header/crystal/alat'][()]            # lattice parameter in bohr
    cell_vol      = f_hdf5['/mf_header/crystal/celvol'][()]      # unit cell vol in bohr**3
    cell_vecs     = f_hdf5['/mf_header/crystal/avec'][()]       # lattice vectors (in units of alat)
    rec_cell_vecs = f_hdf5['/mf_header/crystal/bvec'][()]
    atomic_pos    = f_hdf5['/mf_header/crystal/apos'][()]      # in cartesian coordinates, in units of alat - important for visualization
    Nat           = f_hdf5['/mf_header/crystal/nat'][()]              # Number of atoms
    NQ            = f_hdf5['/exciton_header/kpoints/nQ'][()]               # Number of Q points
    Qshift        = f_hdf5['/exciton_header/kpoints/exciton_Q_shifts'][()]          # Q point shift

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
    # print("!!!!!", ifmax_list.shape)
    
    ifmax_values = []
    for ival in range(ifmax_list.shape[0]):
        if ifmax_values.count(ifmax_list[ival]) == 0:
            ifmax_values.append(ifmax_list[ival])
    
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

    if config["log_k_points"] == True:

        print('Writing k points in eigenvecs in Kpoints_eigenvecs_file')

        arq_kpoints = open('Kpoints_eigenvecs_file', 'w')

        for ik in range(len(Kpoints_bse)):
            kx, ky, kz = Kpoints_bse[ik]
            arq_kpoints.write(f'{kx:.9f}   {ky:.9f}   {kz:.9f}\n')

        arq_kpoints.close()

    # Reporting info from this file

    print(f'\nParameters from {exciton_file} :')

    print(f'    Total of atoms             = {Nat}')
    print(f'    Total of modes vib (3*Nat) = {3*Nat}')
    print(f'    Nkpoints                   = {Nkpoints}')
    print(f'    Number of cond bands       = {Ncbnds}')
    print(f'    Number of val bands        = {Nvbnds}')
    print(f'    Valence band index         = {Nval}')
    print(f'    Number of Q points         = {NQ}')
    print(f'    Q point shift              = {Qshift}')
    if np.linalg.norm(Qshift) > 0.0:
        print(f"This exciton has a finite center of mass momentum")
    print(f'    Lattice parameter (a.u.)   = {alat:.8f}')
    print(f'    Lattice vectors (in lattice parameter units): ')
    print(f'          a1 = ({cell_vecs[0, 0]:.8f}, {cell_vecs[0, 1]:.8f}, {cell_vecs[0, 2]:.8f})')
    print(f'          a2 = ({cell_vecs[1, 0]:.8f}, {cell_vecs[1, 1]:.8f}, {cell_vecs[1, 2]:.8f})')
    print(f'          a3 = ({cell_vecs[2, 0]:.8f}, {cell_vecs[2, 1]:.8f}, {cell_vecs[2, 2]:.8f})')
    print(f'    Reciprocal lattice vectors (2 * pi / lattice parameter):')
    print(f'          b1 = ({rec_cell_vecs[0, 0]:.8f}, {rec_cell_vecs[0, 1]:.8f}, {rec_cell_vecs[0, 2]:.8f})')
    print(f'          b2 = ({rec_cell_vecs[1, 0]:.8f}, {rec_cell_vecs[1, 1]:.8f}, {rec_cell_vecs[1, 2]:.8f})')
    print(f'          b3 = ({rec_cell_vecs[2, 0]:.8f}, {rec_cell_vecs[2, 1]:.8f}, {rec_cell_vecs[2, 2]:.8f})')
    print(f'\n\n')


    return Nat, atomic_pos, cell_vecs, cell_vol, alat, Nvbnds, Ncbnds, Kpoints_bse, Nkpoints, Nval, rec_cell_vecs, NQ, Qshift


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

    #    0.00000   0.00000   0.00000
    #    0.00000   0.00000   0.16667
    #   -0.00000  -0.00000   0.33333
    #   -0.00000   0.00000   0.50000
    #   -0.00000   0.00000  -0.33333


    for line in arq_kpoints_bse:
        linha = line.split()
        kx, ky, kz = float(linha[0]), float(linha[1]), float(linha[2])
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

    # print('Reading exciton info from file', exciton_file)
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

    # print('Exciton energy (eV): '+str(exc_energy)+'\n\n')

    return Akcv #, exc_energy

def top_n_indexes(array, N):
    # Flatten the array
    flat_array = array.flatten()
    
    # Get the indexes of the top N values in the flattened array
    flat_indexes = np.argpartition(flat_array, -N)[-N:]
    
    # Sort these indexes by the values they point to, in descending order
    sorted_indexes = flat_indexes[np.argsort(-flat_array[flat_indexes])]
    
    # Convert the 1D indexes back to 3D indexes
    top_indexes = np.unravel_index(sorted_indexes, array.shape)
    
    # Combine the indexes into a list of tuples
    top_indexes = list(zip(*top_indexes))
    
    return top_indexes

def top_n_indexes_all(array, limit_BSE_sum_up_to_value):
    # Flatten the array
    flat_array = array.flatten()
    
    # array size
    N = len(flat_array)
    
    # Get the indexes of the top N values in the flattened array
    flat_indexes = np.argpartition(flat_array, -N)[-N:]
    
    # Sort these indexes by the values they point to, in descending order
    sorted_indexes = flat_indexes[np.argsort(-flat_array[flat_indexes])]
    
    # Convert the 1D indexes back to 3D indexes
    top_indexes = np.unravel_index(sorted_indexes, array.shape)
    
    # Combine the indexes into a list of tuples
    top_indexes = list(zip(*top_indexes))
    
    # now checking how many values we need to store
    counter_indexes = 0
    sum_abs_Akcv2 = 0
    for index in top_indexes:
        counter_indexes += 1 
        sum_abs_Akcv2 += array[index[0], index[1], index[2]]**2
        if sum_abs_Akcv2 > limit_BSE_sum_up_to_value:
            break

    return top_indexes[:counter_indexes]

def summarize_Acvk(Akcv, BSE_params, limit_BSE_sum_up_to_value):
    
    ''' Print just the relevant information about that exciton. Most of coefficients Acvk
    are null'''
    
    Kpoints_BSE = BSE_params.Kpoints_BSE
    
    print('###############################################')
    print('Showing most relevant coeffs for this exciton')
   
    if limit_BSE_sum_up_to_value == 1.0:
        
        top_indexes = top_n_indexes(np.abs(Akcv), 10)
    else:
        top_indexes = top_n_indexes_all(np.abs(Akcv), limit_BSE_sum_up_to_value)

 
    print('###############################################')
    
    return top_indexes
