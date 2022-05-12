

import numpy as np
import xml.etree.ElementTree as ET
import h5py
import time

def do_I_want_this_band(iband, Nval, N_c_or_v_bnds, c_or_v):

    """
    Checks if a valence (conduction) band is 
    a the one that we need.
    """

    answer = False

    if c_or_v == 'v':  # let's check if it is a valence band
        if Nval - N_c_or_v_bnds < iband <= Nval:
            answer = True

    if c_or_v == 'c':  # let's check if it is a conduction band
        if Nval < iband <= Nval + N_c_or_v_bnds:
            answer = True

    return answer

def indexes_x_in_list(what_i_want, list_i_get):

    """
    Returns the indexes of the value x in a list A
    Ex: A = ['a', 'b', 'a', 'c', 'd', 'd']
    indexes_x_in_list('a', A) returns [0, 2]
    indexes_x_in_list('b', A) returns [1]
    indexes_x_in_list('e', A) returns [] (empty list)
    """

    if list_i_get.count(what_i_want) > 0:
        indexes = [i for i in range(len(list_i_get)) if list_i_get[i] == what_i_want]
    else:
        indexes = []

    return indexes


def get_kernel(kernel_file):

    """
    Reads the kernel matrix elements from BSE calculations
    """

    start_time_func = time.clock_gettime(0)
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

    end_time_func = time.clock_gettime(0)
    print(f'Time spent on get_kernel function: {end_time_func - start_time_func} s')

    return Kd, Kx

def read_eqp_data(eqp_file, Nkpoints, Nvbnds, Ncbnds, Nval):

    """Reads quasiparticle and dft energies results from sigma calculations

    Returns:
        _type_: Eqp_val, Eqp_cond, Edft_val, Edft_cond
    """

    start_time_func = time.clock_gettime(0)

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
            if Nval < int(linha[1]) <= Nval + Ncbnds:
                iband = int(linha[1]) - Nval - 1
                Edft_cond[ik, iband] = float(linha[2])
                Eqp_cond[ik, iband] = float(linha[3])
            if Nvbnds - Nval < int(linha[1]) <= Nval:
                iband = Nval - int(linha[1]) 
                Edft_val[ik, iband] = float(linha[2])
                Eqp_val[ik, iband] = float(linha[3])

    end_time_func = time.clock_gettime(0)
    print(f'Time spent on read_eqp_data function: {end_time_func - start_time_func} s')

    return Eqp_val, Eqp_cond, Edft_val, Edft_cond

def get_exciton_info(exciton_file, Nkpoints, Nvbnds, Ncbnds):

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


def get_patterns2(el_ph_dir, iq, Nmodes, Nat):

    """Reads displacements patterns from patterns.X.xml files, 
    where X is the q vector for this displacement.
    """

    print('Reading displacement patterns file')

    Displacements = np.zeros((Nmodes, Nat, 3), dtype=np.complex64)
    imode = 0
    
    patterns_file = el_ph_dir+'patterns.'+str(iq + 1)+'.xml'
    
    tree = ET.parse(patterns_file)
    root = tree.getroot()

    tags_in_xml_file = [elem.tag for elem in root.iter()]
    texts_in_xml_file = [elem.text for elem in root.iter()]

    # print(tags_in_xml_file)

    Nirreps_index_in_tag = tags_in_xml_file.index('NUMBER_IRR_REP')
    Nirreps = int(texts_in_xml_file[Nirreps_index_in_tag])
    # print(f'Nirreps = {Nirreps}')

    imode = 0

    Npert_indexes_in_tag = indexes_x_in_list('NUMBER_OF_PERTURBATIONS', tags_in_xml_file)
    Displacements_indexes_in_tag = indexes_x_in_list('DISPLACEMENT_PATTERN', tags_in_xml_file)
    # print(Displacements_indexes_in_tag)

    idisp = -1   # counter of displacements

    for irrep in range(Nirreps):

        Npert = int(texts_in_xml_file[Npert_indexes_in_tag[irrep]])
        # print('Npert =', Npert)

        for ipert in range(Npert):
            idisp += 1
            text_temp = texts_in_xml_file[Displacements_indexes_in_tag[idisp]]
            text_temp = text_temp.replace(",", " ")

            numbers_temp = np.fromstring(text_temp, sep='\n')
            
            # reading complex numbers -> A[::2] (A[1::2]) gives the first (second) collum
            temp_displacements = numbers_temp[::2] + 1.0j*numbers_temp[1::2]

            icounter = 0
            for iat in range(Nat):
                for idir in range(3):
                    Displacements[imode, iat, idir] = temp_displacements[icounter]
                    icounter += 1

            imode += 1

    return Displacements, Nirreps


def read_elph_xml(elph_xml_file):

    """Reads elph coefficients (<i_k|dV/dx_mu|j_(k+q)>) produced by DFPT calculations from Quantum Espresso
    Those coefficients are written in .xml files. The nomenclature of those files are
    elph.iq.ipert.xml, where iq = 1,2,3... is the index of the q point that calculation and
    ipert is the index of the perturbation applied for that mode. The displacements patterns 
    for each perturbation are listed in the patterns.iq.xml files.

    Returns:
        elph_aux = [Ndeg, Nkpoints_in_xml_file, Nbnds_in_xml_file, Nbnds_in_xml_file]
        complex array
        where: 
            - Ndeg is the degeneracy of this mode
            - Nkpoints_in_xml is the number of k points in the file
            - Nbnds_in_xml_file is the number of bands in the file 
    """

    print('Reading file ', elph_xml_file)

    tree = ET.parse(elph_xml_file)
    root = tree.getroot()

    # get tags in xml file (ex: NUMBER_OF_BANDS)
    tags_in_xml_file = [elem.tag for elem in root.iter()]
    # get text in xml file for each element. Those are the info we want to read
    texts_in_xml_file = [elem.text for elem in root.iter()]

    # TODO -> just reads 1 k point until now. Needs to be generalized for more q points later

    # reading number of bands in xml file
    Nbnds_index_in_tag = tags_in_xml_file.index('NUMBER_OF_BANDS')
    Nbnds_in_xml_file = int(texts_in_xml_file[Nbnds_index_in_tag])
    print(f'Number of bands in this file {Nbnds_in_xml_file}')

    # reading number of k points in xml file
    Nkpoints_index_in_tag = tags_in_xml_file.index('NUMBER_OF_K')
    Nkpoints_in_xml_file = int(texts_in_xml_file[Nkpoints_index_in_tag])
    print(f'Number of k points in this file {Nkpoints_in_xml_file}')

    # reading elph matrix elements. 
    # print('TAGS', tags_in_xml_file)

    # QE 6.6 and 6.7 report this data in different forms - trying to make it suitable for both versions of QE
    # TODO -> test in which version of QE it works and see what must be done

    # Counting how many times tag 'PARTIAL_ELPH' appears

    how_many_times_tag_ELPH = tags_in_xml_file.count('PARTIAL_ELPH')

    if how_many_times_tag_ELPH == 1:
        elph_mat_elems_index = tags_in_xml_file.index('PARTIAL_ELPH')
        text_temp = texts_in_xml_file[elph_mat_elems_index]
    else:
        text_temp = ''
        for i_tag in range(len(tags_in_xml_file)):
            if tags_in_xml_file[i_tag] == 'PARTIAL_ELPH':
                text_temp += texts_in_xml_file[i_tag]

    # this text is something like this
    # ...
    # 1.1,2.2
    # -3.3,4.4
    # -5.5,6.6
    # ...
    text_temp = text_temp.replace(",", " ")  # replace "," per spaces 
    numbers_temp = np.fromstring(text_temp, sep='\n')  # transforming text to floats

    # Now variable is an 1D array like this
    # ... 1.1 2.2 3.3 4.4 5.5 6.6 ... (same numbers from previous comment)
    # In this array, elements with odd (even) index are the real (imaginary) part 
    # of a complex number. 
    temp_elph = numbers_temp[::2] + 1.0j*numbers_temp[1::2]

    # Now get degeneracy of this mode
    # Number of matrix elements = (Degeneracy of mode) * (Number of bands)**2 
    # print('HELLLOOOO', len(temp_elph), Nbnds_in_xml_file)
    Ndeg = int(len(temp_elph) / Nbnds_in_xml_file**2)
    print(f'Number of modes in this file is {Ndeg}')

    # Building elph matrix
    elph_aux = np.zeros((Ndeg, Nkpoints_in_xml_file, Nbnds_in_xml_file, Nbnds_in_xml_file), dtype=np.complex64)

    contador = 0
    for ideg in range(Ndeg):
        for ibnd in range(Nbnds_in_xml_file):
            for jbnd in range(Nbnds_in_xml_file):
                elph_aux[ideg, 0, ibnd, jbnd] = temp_elph[contador]
                contador += 1

    return elph_aux


def get_el_ph_coeffs(el_ph_dir, iq, Nirreps):  # suitable for xml files written from qe 6.7 

    """ Reads all elph.iq.ipert.xml files and returns the electron-phonon coefficients
    elph[Nmodes, Nk, Nbnds_in_xml, Nbnds_in_xml] """

    print('\n\nReading elph coeficients g_ij = <i|dH/dr|j>\n')

    elph = []

    for irrep in range(Nirreps):
        elph_xml_file = el_ph_dir + f'elph.{iq + 1}.{irrep + 1}.xml'
        elph_aux = read_elph_xml(elph_xml_file)

        for ideg in range(len(elph_aux)):
            elph.append(elph_aux[ideg])

    elph = np.array(elph)

    return elph

def filter_elph_coeffs(elph, params_calc):

    """ Reads elph coefficients from DFPT calculations. Quantum Espresso calculates <i|dV/dx_mu|j> for 
    i, j = 1,2,3,...,Nbnds_in_xml, where Nbnds_in_xml = total of bands included in the scf calculation step before DFPT.
    We just need <c|dV/dx_mu|c'> and <v|dV/dx_mu|v'>, where c ranges from Nval + 1 to Nval + Ncbnds (conduction bands) and
    v ranges from (Nval - Nvbnds + 1) to Nval. In other words, we just need two blocks from the elph matrix.
    
    Other important information, is that the indexes need to be updated as the counting used in the code is different 
    from the one used QE. Conduction bands start been counted from Nval + 1 as 1 upwards and valence bands are counted
    from Nval as 1 downwards. An example is the following:

    indexQE    iv    ic
    1          4
    2          3
    3          2
    4          1
    5                 1
    6                 2
    7                 3

    where the highest valence band (Nval) is 4. The rule to update the indexes are
    iv = Nval - iQE + 1
    ic = iQE - Nval
    """

    Nkpoints, Ncbnds, Nvbnds, Nval, Nmodes = params_calc

    elph_cond = np.zeros((Nmodes, Nkpoints, Ncbnds, Ncbnds), dtype=np.complex64)
    elph_val = np.zeros((Nmodes, Nkpoints, Nvbnds, Nvbnds), dtype=np.complex64)

    Nbnds_in_xml_file = np.shape(elph)[2]
    Ncond_in_xml_file = Nbnds_in_xml_file - Nval

    if Ncond_in_xml_file < Ncbnds:
        print(f'Missing {Ncbnds - Ncond_in_xml_file} cond bands from DFPT calculations. Missing coefficients will be set to 0.')

    for imode in range(Nmodes):
        for ik in range(Nkpoints):

            Ncbnds_to_get = min(Ncbnds, Ncond_in_xml_file)
            Nmin = Nval 
            Nmax = Nval + Ncbnds_to_get 
            elph_cond[imode, ik] = elph[imode, ik, Nmin:Nmax, Nmin:Nmax]

            Nvbnds_to_get = min(Nvbnds, Nval)
            Nmin = Nval - Nvbnds_to_get
            Nmax = Nval 
            temp = elph[imode, ik, Nmin:Nmax, Nmin:Nmax]

            # Making a offdiagonal transpose 
            tuple_iteration = range(-1, -(Nvbnds_to_get+1), -1)  
            elph_val[imode, ik] = np.array([[temp[iv, jv] for iv in tuple_iteration] for jv in tuple_iteration])

    # small report
    print(f"\nMax real value of <c|dH|c'> (eV/A): {np.max(np.real(elph_cond))}")
    print(f"Max imag value of <c|dH|c'> (eV/A): {np.max(np.imag(elph_cond))}")
    print(f"Max real value of <v|dH|v'> (eV/A): {np.max(np.real(elph_val))}")
    print(f"Max imag value of <v|dH|v'> (eV/A): {np.max(np.imag(elph_val))}")

    return elph_cond, elph_val

def get_modes2cart_matrix(dyn_file, Nat, Nmodes):
    # Read eigenvecs - FIXME -> generalize for several q's
    arq = open(dyn_file)

    modes2cart = np.zeros((Nmodes, Nmodes), dtype=np.complex64)

    while True:
        line = arq.readline()
        if '*' in line:
            break

    for imode in range(Nmodes):
        arq.readline()  # freq (    1) = ...
        imodep = 0
        for iat in range(Nat):
            line = arq.readline().split()
            for idir in range(3):
                disp = float(line[1 + 2*idir]) + 1.0j*float(line[2 + 2*idir])
                modes2cart[imodep][imode] = disp
                imodep += 1

    #print(modes2cart)
    arq.close()
    return modes2cart

def calc_DKernel_mat_elem(indexes, Kernel, calc_IBL_way, EDFT, EQP, ELPH, Params, TOL_DEG):

    """Calculates derivatives of kernel matrix elements"""

    # start_time_func = time.clock_gettime(0)

    ik1, ik2, iv1, iv2, ic1, ic2, imode = indexes

    elph_cond, elph_val = ELPH
    Ncbnds, Nvbnds, Nkpoints, Nmodes = Params
    Edft_val, Edft_cond = EDFT

    if calc_IBL_way == True:
        Eqp_val, Eqp_cond = EQP
        DKelement_IBL = 0 + 0.0*1.0j

    DKelement = 0 + 0.0*1.0j
    
    # ik2, ik1, ic2, ic1, iv2, iv1


    for ivp in range(Nvbnds):

        DeltaEdft = Edft_val[ik1, iv1] - Edft_val[ik1, ivp]
        DeltaEqp = Eqp_val[ik1, iv1] - Eqp_val[ik1, ivp]

        indexes_K = ik2, ik1, ic2, ic1, iv2, ivp
        indexes_g = imode, ik1, ivp, iv1

        if abs(DeltaEqp) > TOL_DEG:
            DKelement += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEdft
            if calc_IBL_way == True:
                DKelement_IBL += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEqp

        DeltaEdft = Edft_val[ik2, iv2] - Edft_val[ik2, ivp]
        DeltaEqp = Eqp_val[ik2, iv2] - Eqp_val[ik2, ivp]

        indexes_K = ik2, ik1, ic2, ic1, ivp, iv1
        indexes_g = imode, ik2, iv2, ivp

        if abs(DeltaEqp) > TOL_DEG:
            DKelement += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEdft
            if calc_IBL_way == True:
                DKelement_IBL += Kernel[indexes_K]*elph_val[indexes_g]/DeltaEqp


    for icp in range(Ncbnds):

        DeltaEdft = Edft_cond[ik1, ic1] - Edft_cond[ik1, icp]
        DeltaEqp = Eqp_cond[ik1, ic1] - Eqp_cond[ik1, icp]

        indexes_K = ik2, ik1, ic2, icp, iv2, iv1
        indexes_g = imode, ik1, icp, ic1

        if abs(DeltaEqp) > TOL_DEG:
            DKelement += Kernel[indexes_K]*elph_cond[indexes_g]/DeltaEdft
            if calc_IBL_way == True:
                DKelement_IBL += Kernel[indexes_K]*elph_cond[indexes_g]/DeltaEqp

        DeltaEdft = Edft_cond[ik2, ic2] - Edft_cond[ik2, icp]
        DeltaEqp = Eqp_cond[ik2, ic2] - Eqp_cond[ik2, icp]

        indexes_K = ik2, ik1, icp, ic1, iv2, iv1
        indexes_g = imode, ik2, ic2, icp

        if abs(DeltaEqp) > TOL_DEG:
            DKelement += Kernel[indexes_K]*elph_cond[indexes_g]/DeltaEdft
            if calc_IBL_way == True:
                DKelement_IBL += Kernel[indexes_K]*elph_cond[indexes_g]/DeltaEqp

    # end_time_func = time.clock_gettime(0)
    # print(f'Time spent on calc_DKernel function: {end_time_func - start_time_func} s')

    if calc_IBL_way is True:
        return DKelement, DKelement_IBL
    else:
        return DKelement   


def calc_deriv_Kernel(KernelMat, calc_IBL_way, EDFT, EQP, ELPH, TOL_DEG, Params, Akcv):

    print("    - Calculating Kernel part")

    Ncbnds, Nvbnds, Nkpoints, Nmodes = Params

    Shape2 = (Nmodes, Nkpoints, Ncbnds, Nvbnds, Nkpoints, Ncbnds, Nvbnds)
    DKernel          = np.zeros(Shape2, dtype=np.complex64)
    DKernel_IBL      = np.zeros(Shape2, dtype=np.complex64)

    for imode in range(Nmodes):
        for ik1 in range(Nkpoints):
            for ic1 in range(Ncbnds):
                for iv1 in range(Nvbnds):

                    A_bra = np.conj(Akcv[ik1, ic1, iv1])

                    for ik2 in range(Nkpoints):
                        for ic2 in range(Ncbnds):
                            for iv2 in range(Nvbnds):

                                A_ket = Akcv[ik2, ic2, iv2]

                                indexes = ik1, ik2, iv1, iv2, ic1, ic2, imode
                                dK = calc_DKernel_mat_elem(indexes, KernelMat, calc_IBL_way, EDFT, EQP, ELPH, Params, TOL_DEG)

                                if calc_IBL_way == False:
                                    DKernel[imode, ik1, ic1, iv1, ik2, ic2, iv2] = A_bra * dK * A_ket
                                else:
                                    DKernel[imode, ik1, ic1, iv1, ik2, ic2, iv2] = A_bra * dK[0] * A_ket
                                    DKernel_IBL[imode, ik1, ic1, iv1, ik2, ic2, iv2] = A_bra * dK[1] * A_ket


    if calc_IBL_way == False:
        return DKernel
    else:
        return DKernel, DKernel_IBL


def aux_matrix_elem(Nmodes, Nkpoints, Ncbnds, Nvbnds, elph_cond, elph_val, Edft_val, Edft_cond, Eqp_val, Eqp_cond, TOL_DEG):

    """ Calculates auxiliar matrix elements to be used later in the forces matrix elements.
    Returns aux_cond_matrix, aux_val_matrix and
    aux_cond_matrix[imode, ik, ic1, ic2] = elph_cond[imode, ik, ic1, ic2] * deltaEqp / deltaEdft (if ic1 != ic2)
    aux_val_matrix[imode, ik, iv1, iv2]  = elph_val[imode, ik, iv1, iv2]  * deltaEqp / deltaEdft (if iv1 != iv2)
    If ic1 == ic2 (iv1 == iv2), then the matrix elements are just the elph coefficients"""

    start_time_func = time.clock_gettime(0)

    Shape_cond = (Nmodes, Nkpoints, Ncbnds, Ncbnds)
    aux_cond_matrix = np.zeros(Shape_cond, dtype=np.complex64)

    Shape_val = (Nmodes, Nkpoints, Nvbnds, Nvbnds)
    aux_val_matrix = np.zeros(Shape_val, dtype=np.complex64)

    for imode in range(Nmodes):
        for ik in range(Nkpoints):

            for ic1 in range(Ncbnds):
                for ic2 in range(Ncbnds):

                    elph = elph_cond[imode, ik, ic1, ic2]

                    if ic1 == ic2:
                        aux_cond_matrix[imode, ik, ic1, ic2] = elph
                    
                    elif abs(Edft_cond[ik, ic1] - Edft_cond[ik, ic2]) > TOL_DEG: 
                        deltaEqp = Eqp_cond[ik, ic1] - Eqp_cond[ik, ic2]
                        deltaEdft = Edft_cond[ik, ic1] - Edft_cond[ik, ic2]
                        aux_cond_matrix[imode, ik, ic1, ic2] = elph * deltaEqp / deltaEdft

            for iv1 in range(Nvbnds):
                for iv2 in range(Nvbnds):

                    elph = elph_val[imode, ik, iv1, iv2]

                    if iv1 == iv2:
                        aux_val_matrix[imode, ik, iv1, iv2] = elph
                    
                    elif abs(Edft_val[ik, iv1] - Edft_val[ik, iv2]) > TOL_DEG: 
                        deltaEqp = Eqp_val[ik, iv1] - Eqp_val[ik, iv2]
                        deltaEdft = Edft_val[ik, iv1] - Edft_val[ik, iv2]
                        aux_val_matrix[imode, ik, iv1, iv2] = elph * deltaEqp / deltaEdft
        
    end_time_func = time.clock_gettime(0)
    print(f'Time spent on aux_matrix_elem function: {end_time_func - start_time_func} s')

    return aux_cond_matrix, aux_val_matrix

def delta(i,j):
    """Dirac delta - TODO: check if python has a builtin function that does that."""
    if i == j:
        return 1.0
    else:
        return 0.0

def dirac_delta_Edft(i,j, Edft, TOL_DEG):
    if abs(Edft[0, i] - Edft[0, j]) > TOL_DEG:
        return 1.0
    else:
        return 0.0


def calc_Dkinect_matrix_elem(Akcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2, report_RPA_data):

    """Calculates excited state force matrix elements."""

    Ry2eV = 13.6056980659
    bohr2A = 0.529177249

    # calculate matrix element imode, ik, ic1, ic2, iv1, iv2
    temp_cond = aux_cond_matrix[imode, ik, ic1, ic2]*delta(iv1, iv2)
    temp_val = aux_val_matrix[imode, ik, iv1, iv2]*delta(ic1, ic2)

    tempA = Akcv[ik, ic1, iv1] * np.conj(Akcv[ik, ic2, iv2])
    Dkin = tempA * (temp_cond - temp_val)

    if report_RPA_data == True:
        temp_text = f'{imode+1} {ik+1} {ic1+1} {ic2+1} {iv1+1} {iv2+1} {Dkin} {tempA} {temp_cond} {temp_val} \n'
        return Dkin, temp_text
    
    else:
        return Dkin

def calc_Dkinect_matrix(params_calc, Akcv, aux_cond_matrix, aux_val_matrix, report_RPA_data, just_RPA_diag):

    start_time_func = time.clock_gettime(0)

    Nkpoints, Ncbnds, Nvbnds, Nval, Nmodes = params_calc
    Shape = (Nmodes, Nkpoints, Ncbnds, Nvbnds, Nkpoints, Ncbnds, Nvbnds)
    DKinect = np.zeros(Shape, dtype=np.complex64)

    if report_RPA_data == True:
        arq_RPA_data = open('RPA_matrix_elements.dat', 'w')
        arq_RPA_data.write('# mode ik ic1 ic2 iv1 iv2 F conj(Akc1v1)*Akc2v2 auxMatcond(c1,c2) auxMatval(v1,v2)\n')
        

    if just_RPA_diag == False:
        print('Calculating diag and offdiag RPA force matrix elements')
        for imode in range(Nmodes):
            for ik in range(Nkpoints):
                for ic1 in range(Ncbnds):
                    for ic2 in range(Ncbnds):
                        for iv1 in range(Nvbnds):
                            for iv2 in range(Nvbnds):
                                temp = calc_Dkinect_matrix_elem(Akcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2, report_RPA_data)
                                if report_RPA_data == True:
                                    DKinect[imode, ik, ic1, iv1, ik, ic2, iv2] = temp[0]
                                    arq_RPA_data.write(temp[1])
                                else:
                                    DKinect[imode, ik, ic1, iv1, ik, ic2, iv2] = temp

    else:
        print('Calculating just diag RPA force matrix elements')
        for imode in range(Nmodes):
            for ik in range(Nkpoints):
                for ic1 in range(Ncbnds):
                    for iv1 in range(Nvbnds):
                        temp = calc_Dkinect_matrix_elem(Akcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic1, iv1, iv1, report_RPA_data)
                        if report_RPA_data == True:
                            DKinect[imode, ik, ic1, iv1, ik, ic1, iv1] = temp[0]
                            arq_RPA_data.write(temp[1])
                        else:
                            DKinect[imode, ik, ic1, iv1, ik, ic1, iv1] = temp

    if report_RPA_data == True:
        print(f'RPA matrix elements written in file: RPA_matrix_elements.dat')
        arq_RPA_data.close()

    end_time_func = time.clock_gettime(0)
    print(f'Time spent on calc_Dkinect_matrix function: {end_time_func - start_time_func} s')

    return DKinect
