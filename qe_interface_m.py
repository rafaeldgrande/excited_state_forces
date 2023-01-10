

import time
import xml.etree.ElementTree as ET
import numpy as np
from datetime import datetime


from excited_forces_config import *
from excited_forces_m import *

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


def get_patterns2(iq, MF_params):

    """Reads displacements patterns from patterns.X.xml files, 
    where X is the q vector for this displacement.
    """

    Nat    = MF_params.Nat
    Nmodes = MF_params.Nmodes

    Displacements = np.zeros((Nmodes, Nat, 3))
    imode = 0
    
    patterns_file = el_ph_dir+'patterns.'+str(iq + 1)+'.xml'
    print('\n\nReading displacement patterns file ', patterns_file)
    
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
            #temp_displacements = numbers_temp[::2] + 1.0j*numbers_temp[1::2]
            # displacements are real numbers, so just take the real part
            temp_displacements = numbers_temp[::2] 

            icounter = 0
            for iat in range(Nat):
                for idir in range(3):
                    Displacements[imode, iat, idir] = temp_displacements[icounter]
                    icounter += 1

            imode += 1

    print('Number of irreducible representations = ', Nirreps)

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
    print(f'        Number of bands in this file {Nbnds_in_xml_file}')

    # reading number of k points in xml file
    Nkpoints_index_in_tag = tags_in_xml_file.index('NUMBER_OF_K')
    Nkpoints_in_xml_file = int(texts_in_xml_file[Nkpoints_index_in_tag])
    print(f'        Number of k points in this file {Nkpoints_in_xml_file}')

    # Getting list of k points
    Kpoints_in_elph_file = []
    for i_tag in range(len(tags_in_xml_file)):
        if tags_in_xml_file[i_tag] == 'COORDINATES_XK':
            temp_text = texts_in_xml_file[i_tag].split('\n')[1]
            kx_temp = float(temp_text.split()[0])
            ky_temp = float(temp_text.split()[1])
            kz_temp = float(temp_text.split()[2])
            k_temp  = np.array([kx_temp, ky_temp, kz_temp])
            Kpoints_in_elph_file.append(k_temp)

    if log_k_points == True:

        irrep_name = elph_xml_file.split('.')[-2]  # recovering the irrep name
        # elph_xml_file = something.xxx.yyy.xml -> the above line returns yyy

        arq_kpoints = open('Kpoints_in_elph_file_'+irrep_name, 'w')
        for ik in range(len(Kpoints_in_elph_file)):
            kx, ky, kz = Kpoints_in_elph_file[ik]
            arq_kpoints.write(f'{kx}  {ky}  {kz}  \n')
        arq_kpoints.close()


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
    Ndeg = int( len(temp_elph) / (Nbnds_in_xml_file**2 * Nkpoints_in_xml_file))

    # TODO -> lidar com caso onde tenha degenerescencia E mais de um ponto k
    print(f'        Number of modes in this file is {Ndeg}')

    # Building elph matrix
    elph_aux = np.zeros((Ndeg, Nkpoints_in_xml_file, Nbnds_in_xml_file, Nbnds_in_xml_file), dtype=np.complex64)

    contador = 0
    for ik in range(Nkpoints_in_xml_file):
        for ideg in range(Ndeg):
            for ibnd in range(Nbnds_in_xml_file):
                for jbnd in range(Nbnds_in_xml_file):
                    elph_aux[ideg, ik, ibnd, jbnd] = temp_elph[contador]
                    contador += 1

    return elph_aux, np.array(Kpoints_in_elph_file)


def get_el_ph_coeffs(iq, Nirreps):  # suitable for xml files written from qe 6.7 

    """ Reads all elph.iq.ipert.xml files and returns the electron-phonon coefficients
    elph[Nmodes, Nk, Nbnds_in_xml, Nbnds_in_xml] """

    print('\n\nReading elph coeficients g_ij = <i|dH/dr|j>\n')

    elph = []

    for irrep in range(Nirreps):
        elph_xml_file = el_ph_dir + f'elph.{iq + 1}.{irrep + 1}.xml'
        print('    Reading file ', elph_xml_file, f'({irrep+1}/{Nirreps})')
        elph_aux, Kpoints_in_elph_file = read_elph_xml(elph_xml_file)
        # print('Shape elph_aux', np.shape(elph_aux))

        for ideg in range(len(elph_aux)):
            elph.append(elph_aux[ideg])

    elph = np.array(elph)

    return elph, Kpoints_in_elph_file

def impose_ASR(elph, Displacements, MF_params, acoutic_sum_rule):

    """Impose Acoustic Sum Rule on elph matrix elements
    Test for just CO until now: I know that first and second displacement
    patterns are C and O movements in -z direction respectivelly.
    In future I need to project <i|dH/dr_mu|j> in some direction to remove
    the center of mass translation. Other alternative is to write everything
    in eigenmodes basis, so acoustic modes when q goes to 0 have null el-ph coeffs.
    
    """

    if acoutic_sum_rule == True:

        print('\nApplying acoustic sum rule. Making sum_mu <i|dH/dmu|j> (mu dot n) = 0 for n = x,y,z.')

        Nmodes = MF_params.Nmodes
        Nat    = MF_params.Nat

        mod_sum_report_diag = []
        mod_sum_report_offdiag = []
        mod_sum_report_diag_afterASR = []
        mod_sum_report_offdiag_afterASR = []    

        shape_elph = np.shape(elph)
        Nbnds_in_xml = shape_elph[2]

        for iband1 in range(Nbnds_in_xml):
            for iband2 in range(Nbnds_in_xml):
                # sum_elph = elph[0, 0, iband1, iband2] + elph[1, 0, iband1, iband2]

                # elph[0, 0, iband1, iband2] = elph[0, 0, iband1, iband2] - sum_elph / 2
                # elph[1, 0, iband1, iband2] = elph[1, 0, iband1, iband2] - sum_elph / 2

                sum_elph = np.zeros((3), dtype=complex) # x, y, z

                for i_mode in range(Nmodes):
                    for i_atom in range(Nat):
                        sum_elph += elph[i_mode, 0, iband1, iband2] * Displacements[i_mode, i_atom]

                for i_mode in range(Nmodes):
                    for i_atom in range(Nat):
                        for i_dir in range(3):
                            elph[i_mode, 0, iband1, iband2] = elph[i_mode, 0, iband1, iband2] - Displacements[i_mode, i_atom, i_dir] * sum_elph[i_dir] / Nat

                sum_elph_afterASR = np.zeros((3), dtype=complex) # x, y, z

                for i_mode in range(Nmodes):
                    for i_atom in range(Nat):
                        sum_elph_afterASR += elph[i_mode, 0, iband1, iband2] * Displacements[i_mode, i_atom]


                if iband1 == iband2:
                    for i_dir in range(3):
                        mod_sum_report_diag.append(abs(sum_elph[i_dir]))
                        mod_sum_report_diag_afterASR.append(abs(sum_elph_afterASR[i_dir]))
                else:
                    for i_dir in range(3):
                        mod_sum_report_offdiag.append(abs(sum_elph[i_dir]))
                        mod_sum_report_offdiag_afterASR.append(abs(sum_elph_afterASR[i_dir]))

        mean_val = np.mean(mod_sum_report_diag)
        max_val  = np.max(mod_sum_report_diag)
        mean_val_afterASR = np.mean(mod_sum_report_diag_afterASR)
        max_val_afterASR  = np.max(mod_sum_report_diag_afterASR)
        print("    Mean diag |g_ii| before ASR %.5f" %(mean_val), ' Ry/bohr')
        print("    Max diag  |g_ii| before ASR %.5f" %(max_val), ' Ry/bohr')
        print("    Mean diag |g_ii| after ASR  %.5f" %(mean_val_afterASR), ' Ry/bohr')
        print("    Max diag  |g_ii| after ASR  %.5f" %(max_val_afterASR), ' Ry/bohr')


        mean_val = np.mean(mod_sum_report_offdiag)
        max_val  = np.max(mod_sum_report_offdiag)
        mean_val_afterASR = np.mean(mod_sum_report_offdiag_afterASR)
        max_val_afterASR  = np.max(mod_sum_report_offdiag_afterASR)
        print("    Mean offdiag |g_ij| before ASR %.5f" %(mean_val), ' Ry/bohr')
        print("    Max offdiag  |g_ij| before ASR %.5f" %(max_val), ' Ry/bohr')
        print("    Mean offdiag |g_ij| after ASR  %.5f" %(mean_val_afterASR), ' Ry/bohr')
        print("    Max offdiag  |g_ij| after ASR  %.5f" %(max_val_afterASR), ' Ry/bohr')

    else:
        print('\nNot applying acoustic sum rule')

    return elph

def filter_elph_coeffs(elph, MF_params, BSE_params):

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

    Nmodes   = MF_params.Nmodes
    Nval     = BSE_params.Nval
    
    if elph_fine_a_la_bgw == False:
        Nkpoints     = BSE_params.Nkpoints_BSE
        Ncbnds_sum   = BSE_params.Ncbnds_sum
        Nvbnds_sum   = BSE_params.Nvbnds_sum
    else:
        Nkpoints     = BSE_params.Nkpoints_coarse
        Ncbnds_sum   = BSE_params.Ncbnds_coarse
        Nvbnds_sum   = BSE_params.Nvbnds_coarse

    elph_cond = np.zeros((Nmodes, Nkpoints, Ncbnds_sum, Ncbnds_sum), dtype=np.complex64)
    elph_val = np.zeros((Nmodes, Nkpoints, Nvbnds_sum, Nvbnds_sum), dtype=np.complex64)

    Nbnds_in_xml_file = np.shape(elph)[2]
    Ncond_in_xml_file = Nbnds_in_xml_file - Nval

    if Ncond_in_xml_file < Ncbnds_sum:
        print(f'Missing {Ncbnds_sum - Ncond_in_xml_file} cond bands from DFPT calculations. Missing coefficients will be set to 0.')

    for imode in range(Nmodes):
        for ik in range(Nkpoints):

            Ncbnds_to_get = min(Ncbnds_sum, Ncond_in_xml_file)
            Nmin = Nval 
            Nmax = Nval + Ncbnds_to_get 
            elph_cond[imode, ik] = elph[imode, ik, Nmin:Nmax, Nmin:Nmax]

            Nvbnds_to_get = min(Nvbnds_sum, Nval)
            Nmin = Nval - Nvbnds_to_get
            Nmax = Nval 
            temp = elph[imode, ik, Nmin:Nmax, Nmin:Nmax]

            # Making a offdiagonal transpose 
            tuple_iteration = range(-1, -(Nvbnds_to_get+1), -1)  
            elph_val[imode, ik] = np.array([[temp[iv, jv] for iv in tuple_iteration] for jv in tuple_iteration])

    # small report
    print('\n')
    print("Max real value of <c|dH|c'> (Ry/bohr): %.4f" %(np.max(np.real(elph_cond))))
    print("Max imag value of <c|dH|c'> (Ry/bohr): %.4f" %(np.max(np.imag(elph_cond))))
    print("Max real value of <v|dH|v'> (Ry/bohr): %.4f"  %(np.max(np.real(elph_val))))
    print("Max imag value of <v|dH|v'> (Ry/bohr): %.4f"  %(np.max(np.imag(elph_val))))

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


def elph_interpolate_bgw(elph_co, file_coeffs, Nkpoints_fine, Nbnds_fine):
    
    """
    Make the interpolation of elph (cond or val) coeffs "a la BerkeleyGW".
    
    The BGW code expands wavefunctions in the fine grid in a basis with coarse grid

    fine grid = less bands, more k ponits
    coarse grid = more bands, less k points
    
    The expansion coefficients relate the fine and coarse grid by
    u_(n, k_fi) = sum_(m) C_(n, m)^(k_fi -> k_co) u_(m, k_co)
    
    where u_(n,k) is the periodic part of the Bloch functions.
    
    We have as input:
        - elph coeffs calculated in a coarse grid
        - file with expansion coeffs for bands (dtmat_non_bin_val or dtmat_non_bin_conds)
        - kpoints in the coarse grid (read from kpoints_coarse file. those k points are got from the absorption.out file)
        
    The elph coeffs in a fine grid are given by
    
    g_(ij)^f = sum_(n,m) g_(n,m)^c * C^*_(i,n) * C_(j,m)
    
    where i,j,m,n are val (cond) bands
    i, j are from the fine grid
    n, m are from the coarse grid
    
    The coeffs file looks like this:
    
           1           1           1           1           1
 (0.991465028086625,-0.130372919275044)
           1           2           1           1           1
 (2.537030932414388E-009,1.761237479758188E-009)
           1           3           1           1           1
 (2.364126527817620E-007,4.259280043476210E-007)
           1           4           1           1           1
 (3.354064988511618E-008,-4.543149744118019E-008)
 
    it is written ik_fine iband_fine ik_coarse iband_coarse i_spin (not used now)
 
    The kpoints_coarse file looks like this
    
     0.000000   0.000000   0.000000
     0.000000   0.000000   0.166667
    -0.000000  -0.000000   0.333333
    -0.000000   0.000000   0.500000
    -0.000000   0.000000   0.666667
    -0.000000   0.000000   0.833333
     0.000000   0.166667  -0.000000
     0.000000   0.166667   0.166667
    -0.000000   0.166667   0.333333
    -0.000000   0.166667   0.500000
 
    """
    
    now_this_func = datetime.now()
    
    # np.shape(elph) = (number of modes, number of k points, number of bands, number of bands)
    
    # # reading kpoints_coarse file    
    # kpoints_coarse = np.loadtxt('kpoints_coarse')
    # # each k point -> kpoints_coarse[ik] -> [kx, ky, kz]
    
    # number of val (cond) bands  
    Nbnds_co = np.shape(elph_co)[-1]
    
    # number of modes
    nmodes_elph = np.shape(elph_co)[0]
    
    # number of kpoints in the coarse grid
    Nkpoints_coarse = np.shape(elph_co)[1]
    
    # nkpoints
    elph_fine = np.zeros((nmodes_elph, Nkpoints_fine, Nbnds_fine, Nbnds_fine), dtype=np.complex)
    
    # reading coeffs file
    coeffs = np.zeros((Nkpoints_fine, Nbnds_co, Nbnds_fine), dtype=complex)

    # list translating k points from the fine grid to k points to the coarse grid
    fine_to_coarse = []
    for ik_f in range(Nkpoints_fine):
        fine_to_coarse.append([-1])
    
    # small pre report
    print('    Starting interpolation')
    print(f'    Number of bands in the coarse grid {Nbnds_co}')
    print(f'    Number of bands in the fine grid {Nbnds_fine}')
    print(f'    Number of k points in the coarse grid {Nkpoints_coarse}')
    print(f'    Number of k points in the fine grid {Nkpoints_fine}')
    
    print(f'Reading file {file_coeffs}')
    arq_coeffs = open(file_coeffs)
    
    for line in arq_coeffs:
        line_split = line.split()
        if len(line_split) == 5: # ik_fine iband_fine ik_coarse iband_coarse i_spin (not used now)
            ik_f = int(line_split[0]) - 1
            ib_f = int(line_split[1]) - 1
            ik_c = int(line_split[2]) - 1
            ib_c = int(line_split[3]) - 1
            fine_to_coarse[ik_f] = ik_c
        if len(line_split) == 1:
            temp = line_split[0]
            real_part = float(temp.split(',')[0][1:])
            imaginary_part = float(temp.split(',')[1][:-1])
            coeff_ij = real_part + 1.0j*imaginary_part
            coeffs[ik_f, ib_c, ib_f] = coeff_ij

    # calculating coeffs in the fine grid
    
    print('Starting interpolation')
    
    total_iterations = nmodes_elph * Nkpoints_fine * Nbnds_fine**2 * Nbnds_co**2
    report_interval = step_report(total_iterations)
    counter = 0
    print(f'I will perform {total_iterations} iterations')
    
    for imode in range(nmodes_elph):
        
        for ik_f in range(Nkpoints_fine):
            ik_c = fine_to_coarse[ik_f]
            if ik_c == -1:
                print('WARNING!! Problem at elph coeffs!')
            for ib1_f in range(Nbnds_fine):
                for ib2_f in range(Nbnds_fine):
                    
                    # calculating elph_fine[imode, ik_f, ib1_f, ib2_f]
                    temp_elph_fi = 0 + 0.0j
                    
                    for ib1_c in range(Nbnds_co):
                        for ib2_c in range(Nbnds_co):
                            
                            counter += 1
                            report_iterations(counter, total_iterations, report_interval, now_this_func)
                                                            
                            temp_elph_co = elph_co[imode, ik_c, ib1_c, ib2_c]
                            c_f1_b1 = coeffs[ik_f, ib1_c, ib1_f]
                            c_f2_b2 = coeffs[ik_f, ib2_c, ib2_f]
                                
                            temp_elph_fi += temp_elph_co * np.conj(c_f1_b1) * c_f2_b2
                                
                    elph_fine[imode, ik_f, ib1_f, ib2_f] = temp_elph_fi
                                   
    print('Finished elph interpolation')
    
    return elph_fine
    
    


    
    
    