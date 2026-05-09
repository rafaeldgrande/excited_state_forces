
from excited_forces_config import *
from modules_to_import import *

ry2ev = 13.605698066

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
    
    print(f'\n\nReading DFT and QP energies from file {eqp_file}')
    
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
                        
    print(f'Finished reading energies from file {eqp_file}')
    print('QP and DFT energy levels for cond bands have shape: ', Eqp_cond.shape, ' = (Nkpoints, Ncbnds)')
    print('QP and DFT energy levels for val bands have shape: ', Eqp_val.shape, ' = (Nkpoints, Nvbnds)')
    print('\n\n')
    
    return Eqp_val, Eqp_cond, Edft_val, Edft_cond


################ BSE related functions ####################

def copy_hdf5_dataset(file_path, dataset_name):
    """
    Reads and copies a dataset from an HDF5 file into a NumPy array.
    Args:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset to copy.
    Returns:
        np.ndarray: A NumPy array containing the dataset.
    """
    with h5py.File(file_path, 'r') as f:
        if dataset_name in f:
            data = np.array(f[dataset_name])  # Copy dataset to a NumPy array
            print(f"Dataset '{dataset_name}' copied from file {file_path}. Shape: {data.shape}, Dtype: {data.dtype}")
            return data
        else:
            print(f"Dataset '{dataset_name}' not found in the file.")
            return None

def reverse_bse_index(ibse, Nk, Nc, Nv, Nspin=1):
    """
    Given a BSE index, return the corresponding k-point (ik), conduction band index (ic),
    and valence band index (iv) assuming ispin = 0 (Python indexing).

    Args:
        ibse (int): Flattened BSE index.
        Nk (int): Total number of k-points.
        Nc (int): Total number of conduction bands.
        Nv (int): Total number of valence bands.
        Nspin (int, optional): Number of spin channels (default is 1).

    Returns:
        tuple: (ik, ic, iv) corresponding to the given BSE index.
    """
    temp = ibse // Nspin  # Remove spin dependence

    iv = temp % Nv
    temp //= Nv

    ic = temp % Nc
    temp //= Nc

    ik = temp  # Remaining value is ik

    return ik, ic, iv

def load_hbse_matrix(hbse_file, Nkpoints_BSE, Ncbnds, Nvbnds):

    print("Loading hbse matrix from file", hbse_file)
    hbse_matrix_temp = copy_hdf5_dataset(hbse_file, "hbse_a")
    hbse_matrix_temp = ry2ev * (hbse_matrix_temp[:,:,0] + 1.0j*hbse_matrix_temp[:,:,1])
    hbse = np.zeros((Nkpoints_BSE, Ncbnds, Nvbnds, Nkpoints_BSE, Ncbnds, Nvbnds), dtype=complex)
    Nexc = hbse_matrix_temp.shape[0]
    for iexc1 in range(Nexc):
        ik1, ic1, iv1 = reverse_bse_index(iexc1, Nkpoints_BSE, Ncbnds, Nvbnds)
        for iexc2 in range(Nexc):
            ik2, ic2, iv2 = reverse_bse_index(iexc2, Nkpoints_BSE, Ncbnds, Nvbnds)
            hbse[ik1, ic1, iv1, ik2, ic2, iv2] = hbse_matrix_temp[iexc2, iexc1]
    
    print('Original shape ', hbse_matrix_temp.shape)
    print('New shape ', hbse.shape)   
      
    return hbse

def rpa_part_from_eqp(Eqp_cond, Eqp_val):
    Nkpoints_BSE, Ncbnds = Eqp_cond.shape
    _, Nvbnds = Eqp_val.shape
    rpa_part = np.zeros((Nkpoints_BSE, Ncbnds, Nvbnds, Nkpoints_BSE, Ncbnds, Nvbnds), dtype=float)
    for ik in range(Nkpoints_BSE):
        for ic in range(Ncbnds):
            for iv in range(Nvbnds):
                rpa_part[ik, ic, iv, ik, ic, iv] = Eqp_cond[ik, ic] - Eqp_val[ik, iv]
    return rpa_part
    
def get_kernel_from_hbse(hbse_file, Eqp_cond, Eqp_val):
    
    Nkpoints_BSE, Ncbnds = Eqp_cond.shape
    _, Nvbnds = Eqp_val.shape
    # print('get_kernel_from_hbse: Nkpoints_BSE, Ncbnds, Nvbnds = ', Nkpoints_BSE, Ncbnds, Nvbnds)
    
    hbse = load_hbse_matrix(hbse_file, Nkpoints_BSE, Ncbnds, Nvbnds) # units Ry, so converting to eV
    rpa_part = rpa_part_from_eqp(Eqp_cond, Eqp_val) # units eV
    
    # print('hbse shape ', hbse.shape)
    # print('rpa_part shape ', rpa_part.shape)
    
    return hbse - rpa_part




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

def load_excitons_coefficients(exciton_file, excitons_to_be_loaded):

    excitons_to_be_loaded_ini_0 = [iexc - 1 for iexc in excitons_to_be_loaded]  # convert to ini=0
    f_hdf5 = h5py.File(exciton_file, 'r')
    
    flavor_calc = f_hdf5['/exciton_header/flavor'][()]
    eigenvecs   = f_hdf5['exciton_data/eigenvectors'][()] # (nQ, Nevecs, nk, nc, nv, ns, real or imag part)
    eigenvalues   = f_hdf5['exciton_data/eigenvalues'][()]   # (Nevecs)

    if flavor_calc == 2:
        Akcv = eigenvecs[0, excitons_to_be_loaded_ini_0,:,:,:,0,0] + 1.0j*eigenvecs[0, excitons_to_be_loaded_ini_0,:,:,:,0,1]
    else:
        Akcv = eigenvecs[0, excitons_to_be_loaded_ini_0,:,:,:,0,0]
        
    for iexc in excitons_to_be_loaded_ini_0:
        print(f'Exciton {iexc+1} energy (eV) = {eigenvalues[iexc]:.6f}')

    eigenvalues_loaded = eigenvalues[excitons_to_be_loaded_ini_0]
    if np.any(np.isnan(eigenvalues_loaded)):
        nan_list = [excitons_to_be_loaded_ini_0[i] + 1
                    for i in range(len(eigenvalues_loaded))
                    if np.isnan(eigenvalues_loaded[i])]
        print(f'  WARNING: NaN eigenvalue(s) for exciton(s) {nan_list} — '
              f'energies unavailable in this file (force calculation is unaffected).')
    return Akcv, eigenvalues_loaded


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
        print(f' ifmax through k points is just one value ({ifmax_values[0]})')
        print(f" so we are dealing probably with a semiconductor. No warning regarging this.")
    else:
        print('######################################################\n')
        print(f'WARNING! ifmax changes through k points! It means that the system is metallic, and we STILL did not implement it.')
        print(f'I will work with it as a semiconductor by setting the valence band to be min(ifmax) = {Nval}')
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

