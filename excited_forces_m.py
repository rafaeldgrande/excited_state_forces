

import numpy as np
import xml.etree.ElementTree as ET
import h5py


def do_I_want_this_band(iband, Nval, N_c_or_v_bnds, c_or_v):

    answer = False

    if c_or_v == 'v':  # let's check if it is a valence band
        if Nval - N_c_or_v_bnds < iband <= Nval:
            answer = True

    if c_or_v == 'c':  # let's check if it is a conduction band
        if Nval < iband <= Nval + N_c_or_v_bnds:
            answer = True

    return answer

def get_kernel(kernel_file):

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

    return Kd, Kx

def read_eqp_data(eqp_file, Nkpoints, Nvbnds, Ncbnds, Nval):

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

    return Eqp_val, Eqp_cond, Edft_val, Edft_cond

def get_exciton_info(exciton_file, Nkpoints, Nvbnds, Ncbnds):

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


def get_patterns2(el_ph_dir, iq, Nmodes, Nat):

    Displacements = np.zeros((Nmodes, Nat, 3), dtype=np.complex64)
    imode = 0
    Perts = []
    
    patterns_file = el_ph_dir+'patterns.'+str(iq + 1)+'.xml'
    
    tree = ET.parse(patterns_file)
    root = tree.getroot()

    #Nirreps = int(root[0][3].text.split("\n")[1])
    Nirreps = int(root[0][3].text)

    imode = 0
    for irreps in range(Nirreps):

        # number of perturbations for this representation
        #n_pert = int(root[0][irreps + 4][0].text.split('\n')[1])
        Npert = int(root[0][irreps + 4][0].text)
        Perts.append(Npert)

        for ipert in range(Npert):
            #text_temp = root[0][irreps + 4][1][ipert + 2].text
            text_temp = root[0][irreps + 4][1 + ipert][0].text
            print("Olha aqui!", text_temp, irreps, ipert)
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

    return Displacements, Nirreps, Perts




def get_patterns(el_ph_dir, iq, Nmodes, Nat):  # suitable for xml written from qe 6.7 
    # total of pattern files is Nq and
    # nomenclature is  patterns.iq.xml with 1 <= iq <= Nq

    Displacements = np.zeros((Nmodes, Nat, 3), dtype=np.complex64)
    imode = 0
    Perts = []
    
    patterns_file = el_ph_dir+'patterns.'+str(iq + 1)+'.xml'
    arq = open(patterns_file)

    for i in range(6):  # do not care about this info now
        arq.readline()

    line = arq.readline()  # <NUMBER_IRR_REP>XX</NUMBER_IRR_REP>
    Nirreps = int(line.split('<')[1].split('>')[1])

    for i_irreps in range(Nirreps):
        arq.readline()  # <REPRESENTION.X>

        line = arq.readline() # <NUMBER_OF_PERTURBATIONS>1</NUMBER_OF_PERTURBATIONS>

        Npert = int(line.split('<')[1].split('>')[1])
        Perts.append(Npert)

        for ipert in range(Npert):

            arq.readline() # <PERTURBATION.1>
            arq.readline() # <DISPLACEMENT_PATTERN>

            for iat in range(Nat):  # reading 3*Nat displacements - it is a complex number
                for idir in range(3):
                    line = arq.readline().split()  
                    Displacements[imode][iat][idir] = float(line[0]) + 1.0j*float(line[1])

            arq.readline() # </DISPLACEMENT_PATTERN>
            arq.readline() # </PERTURBATION.1>

            imode += 1

        arq.readline()  # </REPRESENTION.1>

    arq.close()

    return Displacements, Nirreps, Perts        

def get_el_ph_coeffs2(el_ph_dir, iq, Nirreps, params_calc):  # suitable for xml written from qe 6.7 

    print('Reading elph coeficients g_ij = <i|dH/dr|j>\n')

    Nkpoints, Ncbnds, Nvbnds, Nval, Nmodes = params_calc

    Nbnds = Nvbnds + Ncbnds
    elph_cond = np.zeros((Nmodes, Nkpoints, Ncbnds, Ncbnds), dtype=np.complex64)
    elph_val = np.zeros((Nmodes, Nkpoints, Nvbnds, Nvbnds), dtype=np.complex64)

    # How many bnds in xml files?
    
    el_ph_coeffs_file = el_ph_dir+'elph.'+str(iq + 1)+'.1.xml'

    tree = ET.parse(el_ph_coeffs_file)
    root = tree.getroot()
    Nbnds_in_xml_file = int(root[1][1].text)

    # para checagem
    elph_aux = np.zeros((Nmodes, Nkpoints, Nbnds_in_xml_file, Nbnds_in_xml_file), dtype=np.complex64)

    if Nbnds > Nbnds_in_xml_file:
        print('WARNING! Number of bands in elph files is not enough!')
        print('Setting missing elph coefficients to 0. '+ str(Nbnds - Nbnds_in_xml_file)+' bands are missing')

    imode = 0 # contador de modos

    for i_irrep in range(Nirreps):

        el_ph_coeffs_file = el_ph_dir+'elph.'+str(iq + 1)+'.'+str(i_irrep+1)+'.xml'
        print('Reading: ', el_ph_coeffs_file)

        tree = ET.parse(el_ph_coeffs_file)
        root = tree.getroot()

        Npert = len(root[1][2]) - 1
        print('This file has '+str(Npert)+' modes')

        for ipert in range(Npert):
            for ik in range(Nkpoints):    
                text_temp = root[1][2 + ik][ipert + 1].text
                text_temp = text_temp.replace(",", " ")
                numbers_temp = np.fromstring(text_temp, sep='\n')
                # reading complex numbers -> A[::2] (A[1::2]) gives the first (second) collum
                temp_elph = numbers_temp[::2] + 1.0j*numbers_temp[1::2]

                icounter = 0
                for ibnd in range(Nbnds_in_xml_file):
                    for jbnd in range(Nbnds_in_xml_file):
                        elph_aux[imode, ik, ibnd - Nval, jbnd - Nval] = temp_elph[icounter]

                        if do_I_want_this_band(ibnd, Nval, Nvbnds, 'v') == True and do_I_want_this_band(jbnd, Nval, Nvbnds, 'v') == True:
                            ibndp, jbndp = Nval - ibnd, Nval - jbnd
                            elph_val[imode, ik, ibndp, jbndp] = temp_elph[icounter]
                        
                        if do_I_want_this_band(ibnd, Nval, Ncbnds, 'c') == True and do_I_want_this_band(jbnd, Nval, Ncbnds, 'c') == True:
                            ibndp, jbndp = Nval - ibnd, Nval - jbnd
                            elph_cond[imode, ik, ibndp, jbndp] = temp_elph[icounter]

                        icounter += 1
            imode += 1

    return elph_aux, elph_cond, elph_val


def filter_elph_coeffs(elph_aux, Ncbnds, Nvbnds, Nkpoints, Nmodes, Nval):

    elph_cond = np.zeros((Nmodes, Nkpoints, Ncbnds, Ncbnds), dtype=np.complex64)
    elph_val = np.zeros((Nmodes, Nkpoints, Nvbnds, Nvbnds), dtype=np.complex64)

    for imode in range(Nmodes):
        for ik in range(Nkpoints):

            for iv1 in range(Nvbnds):
                ivp1 = (Nval - 1) - iv1
                for iv2 in range(Nvbnds):
                    ivp2 = (Nval - 1) - iv2
                    elph_val[imode][ik][iv1][iv2] = elph_aux[imode][ik][ivp1][ivp2]

            for ic1 in range(Ncbnds):
                icp1 = Nval + ic1
                for ic2 in range(Ncbnds):
                    icp2 = Nval + ic2
                    elph_cond[imode][ik][ic1][ic2] = elph_aux[imode][ik][icp1][icp2]

    return elph_cond, elph_val


def get_el_ph_coeffs(el_ph_dir, iq, Nirreps, Perts, Nmodes, Nkpoints, Ncbnds, Nvbnds, Nval):  # suitable for xml written from qe 6.7 
    # FIXME -> rewrite it to read .xml files using .xml modules in python
    # total of el_ph files is Nq*Nirreps and 
    # nomenclature is elph."iq"."i_irrep".xml with 
    # 1 <= iq <= Nq and 1 <= i_irrep <= Nirreps
    # Here I will read all irreps for a given iq

    Nbnds = Nvbnds + Ncbnds

    elph_cond = np.zeros((Nmodes, Nkpoints, Ncbnds, Ncbnds), dtype=np.complex64)
    elph_val = np.zeros((Nmodes, Nkpoints, Nvbnds, Nvbnds), dtype=np.complex64) 
    elph_aux = []
    # I just want <c'k|dV/du_{mode, q=0}|ck>  and 
    # <v'k|dV/du_{mode, q=0}|vk>, so I will read the file 
    # entirely and then filter to return those quantities

    imode = -1 # contador de modos

    for i_irrep in range(Nirreps):

        el_ph_coeffs_file = el_ph_dir+'elph.'+str(iq + 1)+'.'+str(i_irrep+1)+'.xml'
        print('Reading: ', el_ph_coeffs_file)
        

        arq = open(el_ph_coeffs_file)

        for i in range(6):  # not important for me
            arq.readline()

        # <NUMBER_OF_BANDS>31</NUMBER_OF_BANDS>
        temp_line = arq.readline()
        temp_Nbnds = int(temp_line.split("<")[1].split(">")[1])
        print('Number of bands in .xml file '+str(temp_Nbnds))
        elph_aux_temp = np.zeros((Nkpoints, temp_Nbnds, temp_Nbnds), dtype=np.complex64)

        for ik in range(Nkpoints):

            arq.readline() # <K_POINT.1>

            for i in range(3): # not important for me
                arq.readline()

            for ipert in range(Perts[i_irrep]):
                imode += 1
                print('    Now in mode ', imode + 1)
                arq.readline()  # <PARTIAL_ELPH perturbation="1">

                for iband in range(temp_Nbnds):
                    for jband in range(temp_Nbnds):
                        line = arq.readline().split()
                        #print(iband, jband, line)
                        elph_aux_temp[ik][iband][jband] = float(line[0]) + 1.0j*float(line[1])

                arq.readline()  # </PARTIAL_ELPH>
                elph_aux.append(elph_aux_temp)

            arq.readline() # </K_POINT.1>
        
        arq.close()

    # convert from list to array
    elph_aux = np.array(elph_aux)
    # filtering data

    for imode in range(Nmodes):
        for ik in range(Nkpoints):

            for iband_new in range(Ncbnds):
                iband = iband_new + Nval
                for jband_new in range(Ncbnds):
                    jband = jband_new + Nval
                    elph_cond[imode][ik][iband_new][jband_new] = elph_aux[imode][ik][iband][jband]

            for iband_new in range(Nvbnds):
                iband = Nval - 1 - iband_new
                for jband_new in range(Nvbnds):
                    jband = Nval - 1 - jband_new
                    elph_val[imode][ik][iband_new][jband_new] = elph_aux[imode][ik][iband][jband]

    return elph_val, elph_cond, elph_aux

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

def calc_DKernel(indexes, Kernel, calc_IBL_way, EDFT, EQP, ELPH, Nparams, TOL_DEG):

    ik1, ik2, iv1, iv2, ic1, ic2, imode = indexes

    elph_cond, elph_val = ELPH
    Ncbnds, Nvbnds, Nkpoints = Nparams
    Edft_val, Edft_cond = EDFT

    if calc_IBL_way == True:
        Eqp_val, Eqp_cond = EQP
        DKelement_IBL = 0 + 0.0*1.0j

    DKelement = 0 + 0.0*1.0j
    
    for ikp in range(Nkpoints):

        for ivp in range(Nvbnds):

            DeltaEdft = Edft_val[ik1][iv1] - Edft_val[ikp][ivp]
            if calc_IBL_way == True:
                DeltaEqp = Eqp_val[ik1][iv1] - Eqp_val[ikp][ivp]
            if abs(DeltaEdft) > TOL_DEG:
                DKelement += Kernel[ikp][ik2][ic1][ic2][ivp][iv2]*elph_val[imode][ikp][ivp][iv1]/DeltaEdft
                if calc_IBL_way == True:
                    DKelement_IBL += Kernel[ikp][ik2][ic1][ic2][ivp][iv2]*elph_val[imode][ikp][ivp][iv1]/DeltaEqp

            DeltaEdft = Edft_val[ik2][iv2] - Edft_val[ikp][ivp]
            if calc_IBL_way == True:
                DeltaEqp = Eqp_val[ik2][iv2] - Eqp_val[ikp][ivp]
            if abs(DeltaEdft) > TOL_DEG:
                DKelement += Kernel[ik1][ikp][ic1][ic2][iv1][ivp]*elph_val[imode][ikp][iv1][ivp]/DeltaEdft
                if calc_IBL_way == True:
                    DKelement_IBL += Kernel[ik1][ikp][ic1][ic2][iv1][ivp]*elph_val[imode][ikp][iv1][ivp]/DeltaEqp

        for icp in range(Ncbnds):

            DeltaEdft = Edft_cond[ik1][ic1] - Edft_val[ikp][ivp]
            if calc_IBL_way == True:
                DeltaEqp = Eqp_cond[ik1][ic1] - Eqp_val[ikp][ivp]
            if abs(DeltaEdft) > TOL_DEG:
                DKelement += Kernel[ikp][ik2][icp][ic2][iv1][iv2]*elph_cond[imode][ikp][ic1][icp]/DeltaEdft
                if calc_IBL_way == True:
                    DKelement_IBL += Kernel[ikp][ik2][icp][ic2][iv1][iv2]*elph_cond[imode][ikp][ic1][icp]/DeltaEqp

            DeltaEdft = Edft_cond[ik2][ic2] - Edft_cond[ikp][icp]
            if calc_IBL_way == True:
                DeltaEqp = Eqp_cond[ik2][ic2] - Eqp_cond[ikp][icp]
            if abs(DeltaEdft) > TOL_DEG:
                DKelement += Kernel[ik1][ikp][ic1][icp][iv1][iv2]*elph_cond[imode][ikp][icp][ic2]/DeltaEdft
                if calc_IBL_way == True:
                    DKelement_IBL += Kernel[ik1][ikp][ic1][icp][iv1][iv2]*elph_cond[imode][ikp][icp][ic2]/DeltaEqp


    if calc_IBL_way is True:
        return DKelement, DKelement_IBL
    else:
        return DKelement   


def calculate_Fcvk(Ncbnds, Nvbnds, Akcv, Edft_cond, Edft_val, Eqp_cond, Eqp_val, elph_cond, elph_val, imode, ik, ic, iv, TOL_DEG):

    Fcvk_offdiag = 0.0 + 1.0j*0.0
    Fcvk_diag = 0.0 + 1.0j*0.0

    for icp in range(Ncbnds):
        temp1 = elph_cond[imode, ik, icp, ic]*Akcv[ik, icp, iv]
        #print('DeltaE ik, ic, icp', ik, ic, icp, abs(Edft_cond[ik, icp] - Edft_cond[ik, ic]))
        if icp == ic:
            Fcvk_diag += temp1
        elif abs(Edft_cond[ik, icp] - Edft_cond[ik, ic]) > TOL_DEG:            
            temp2 = (Eqp_cond[ik, icp] - Eqp_cond[ik, ic]) / (Edft_cond[ik, icp] - Edft_cond[ik, ic])
            Fcvk_offdiag += temp1*temp2
            #print('debug cond - icp, ic, iv,', icp, ic, iv, temp2*temp1)

    for ivp in range(Nvbnds):
        temp1 = elph_val[imode, ik, ivp, iv]*Akcv[ik, ic, ivp]
        if ivp == iv:
            Fcvk_diag -= temp1
        elif abs(Edft_val[ik, ivp] - Edft_val[ik, iv]) > TOL_DEG:
            temp2 = (Eqp_val[ik, ivp] - Eqp_val[ik, iv]) / (Edft_val[ik, ivp] - Edft_val[ik, iv])
            Fcvk_offdiag -= temp1*temp2
            #print('debug val - ivp, iv, ic,', ivp, iv, ic, temp2*temp1)
    print('DeltaE ik, ic, iv', ik, ic, iv,Fcvk_diag, Fcvk_offdiag)
    #print(ik, ic, iv, Fcvk_diag, Fcvk_offdiag)
    return Fcvk_diag, Fcvk_offdiag


def aux_matrix_elem(Nmodes, Nkpoints, Ncbnds, Nvbnds, elph_cond, elph_val, Edft_val, Edft_cond, Eqp_val, Eqp_cond, TOL_DEG):

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
                        aux_cond_matrix[imode][ik][ic1][ic2] = elph
                    
                    elif abs(Edft_cond[ik, ic1] - Edft_cond[ik, ic2]) > TOL_DEG: 
                        deltaEqp = Eqp_cond[ik, ic1] - Eqp_cond[ik, ic2]
                        deltaEdft = Edft_cond[ik, ic1] - Edft_cond[ik, ic2]
                        aux_cond_matrix[imode, ik, ic1, ic2] = elph * deltaEqp / deltaEdft

            for iv1 in range(Nvbnds):
                for iv2 in range(Nvbnds):

                    elph = elph_val[imode, ik, iv1, iv2]

                    if iv1 == iv2:
                        aux_val_matrix[imode][ik][iv1][iv2] = elph
                    
                    elif abs(Edft_val[ik, iv1] - Edft_val[ik, iv2]) > TOL_DEG: 
                        deltaEqp = Eqp_val[ik, iv1] - Eqp_val[ik, iv2]
                        deltaEdft = Edft_val[ik, iv1] - Edft_val[ik, iv2]
                        aux_val_matrix[imode, ik, iv1, iv2] = elph * deltaEqp / deltaEdft
        
    return aux_cond_matrix, aux_val_matrix

def calc_Dkinect_matrix_elem(Akcv, aux_cond_matrix, aux_val_matrix, imode, ik, ic1, ic2, iv1, iv2):

    # calculate matrix element imode, ik, ic1, ic2, iv1, iv2
    D_c1v1k_c2v2k = aux_cond_matrix[imode, ik, ic1, ic2] - aux_val_matrix[imode, ik, iv1, iv2]
    return Akcv[ik, ic1, iv1] * np.conj(Akcv[ik, ic2, iv2]) * D_c1v1k_c2v2k