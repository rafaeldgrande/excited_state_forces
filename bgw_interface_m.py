
import h5py
import time
import numpy as np

from excited_forces_config import *


################ GW related functions #####################

def read_eqp_data(eqp_file):

    """Reads quasiparticle and dft energies results from sigma calculations

    Returns:
        _type_: Eqp_val, Eqp_cond, Edft_val, Edft_cond
    """

    Eqp_val   = np.zeros((Nkpoints, Nvbnds), dtype=np.float64)
    Edft_val  = np.zeros((Nkpoints, Nvbnds), dtype=np.float64)
    Eqp_cond  = np.zeros((Nkpoints, Ncbnds), dtype=np.float64)
    Edft_cond = np.zeros((Nkpoints, Ncbnds), dtype=np.float64)

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

def get_kernel(kernel_file):

    """
    Reads the kernel matrix elements from BSE calculations
    """

    # start_time_func = time.clock_gettime(0)
    print('\nReading kernel matrix elements from ', kernel_file)

    # Kd = head (G=G'=0) + wing (G=0 or G'=0) + body (otherwise) - see https://doi.org/10.1016/j.cpc.2011.12.006

    f_hdf5 = h5py.File(kernel_file, 'r')

    Head = f_hdf5['mats/head']
    Body = f_hdf5['mats/body']
    Wing = f_hdf5['mats/wing']
    Exchange = f_hdf5['mats/exchange']

    Kd = Head[:,:,:,:,:,:,0] + 1.0j*Head[:,:,:,:,:,:,1]
    Kd += Wing[:,:,:,:,:,:,0] + 1.0j*Wing[:,:,:,:,:,:,1]
    Kd += Body[:,:,:,:,:,:,0] + 1.0j*Body[:,:,:,:,:,:,1]

    Kx = Exchange[:,:,:,:,:,:,0] + 1.0j*Exchange[:,:,:,:,:,:,1]

    # end_time_func = time.clock_gettime(0)
    # print(f'Time spent on get_kernel function: '+report_time(start_time_func))

    return Kd, Kx


def get_exciton_info(exciton_file):

    """
    When eigenvectors.h5 files are not available, must use this alternative here
    Have to use my modified version of summarize_eigenvectors code from BGW
    https://github.com/rafaeldgrande/utilities/blob/main/BGW/modified_summarize_eigenvectors.f90
    """

    Akcv = np.zeros((Nkpoints, Ncbnds, Nvbnds), dtype=np.complex64)

    print('Reading exciton info from file', exciton_file)
    arq = open(exciton_file)

    for line in arq:
        linha = line.split()
        if len(linha) == 6:
            if linha[0] != 'Special' and linha[0] != 'c':
                ic, iv, ik = int(linha[0]) - 1, int(linha[1]) - 1, int(linha[2]) - 1
                if ic < Ncbnds and iv < Nvbnds:
                    Akcv[ik][ic][iv] = float(linha[3]) + 1.0j*float(linha[4])
            if linha[0] == 'Special':
                exc_energy = float(linha[-1])

    print('Exciton energy (eV): '+str(exc_energy)+'\n\n')

    return Akcv, exc_energy

def get_hdf5_exciton_info(exciton_file, iexc):

    """    
    Return the exciton energy and the eigenvec coefficients Acvk

    Assuming calculations with TD approximation
    Info about file at: http://manual.berkeleygw.org/3.0/eigenvectors_h5_spec/
    Also, just working for excitons with Q = 0

    TODO -> for now calculting exciton info for exciton with index iexc
    but later, make it calculate for and set of exciton indexes"""

    print('Reading exciton info from file', exciton_file)

    f_hdf5 = h5py.File(exciton_file, 'r')

    eigenvecs = f_hdf5['exciton_data/eigenvectors']
    eigenvals = f_hdf5['exciton_data/eigenvalues']

    Acvk = eigenvecs[0,iexc-1,:,:,:,:,0] + 1.0j*eigenvecs[0,iexc-1,:,:,:,:,1]
    Omega = eigenvals[iexc]

    return Acvk, Omega

