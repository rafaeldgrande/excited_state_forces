
import numpy as np

# File with variables used in excited_forces.py code
# It reads input forces.inp file
# If file not found or some parameter not find in file
# then the code uses (dumb) default values



TOL_DEG = 1e-5  # tolerance to see if two energy values are degenerate
Ry2eV = 13.6056980659
bohr2A = 0.529177249

# Dumb default parameters
iexc = 1
# # files and paths to be opened 
eqp_file = 'eqp1.dat'
exciton_file = 'eigenvectors.h5'
el_ph_dir = './'
kernel_file = 'bsemat.h5'

# conditionals
calc_modes_basis = False    # not being used yet 

calc_IBL_way = True         # I should comment later
write_DKernel = False       # not being used  yet
Calculate_Kernel = False    # Dont change. We dont know how to work with kernel yet

just_RPA_diag = False       # If true doesn't calculate forces a la David
report_RPA_data = False     # report Fcvkc'v'k' matrix elements.

read_Akcv_trick = False     # If false, read eigenvectors.h5 file, when BGW is compiled with hdf5. If true read Acvk{iexc} that is an output of my modified summarize_eigenvectors.f90 file

show_imag_part = False      # Show imaginary part of excited state force . It should be very close to 0 

acoutic_sum_rule = True     # Makes the excited state forces to sum to zero (obey Newton's third law), by making sum of elph matrix elems to 0. May want to set this variable to true
use_hermicity_F = True      # Use the fact that F_cvc'v' = conj(F_c'v'cv)
                            # Reduces the number of computed terms by about half

log_k_points = True          # Write k points used in BSE and DFPT calculations


def true_or_false(text, default_value):
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False
    else:
        return default_value

def read_input(input_file):

    # getting necessary info

    global iexc
    global eqp_file, exciton_file, el_ph_dir
    global dyn_file, kernel_file
    global calc_modes_basis
    global calc_IBL_way, write_DKernel, Calculate_Kernel
    global just_RPA_diag, report_RPA_data
    global show_imag_part
    global read_Akcv_trick
    global acoutic_sum_rule, use_hermicity_F
    global log_k_points

    try:
        arq_in = open(input_file)
        print(f'Reading input file {input_file}')
        did_I_find_the_file = True
    except:
        did_I_find_the_file = False
        print(f'WARNING! - Input file {input_file} not found!')
        print('Using default values for configuration')

    if did_I_find_the_file == True:
        for line in arq_in:
            linha = line.split()
            if len(linha) >= 2:

######## parameters and files ######################

                if linha[0] == 'iexc':
                    iexc = int(linha[1])
                elif linha[0] == 'eqp_file':
                    eqp_file = linha[1]
                elif linha[0] == 'exciton_file':
                    exciton_file = linha[1]
                elif linha[0] == 'el_ph_dir':
                    el_ph_dir = linha[1]
                elif linha[0] == 'dyn_file':
                    dyn_file = linha[1]
                elif linha[0] == 'kernel_file':
                    kernel_file = linha[1]
######### conditionals #############################

                elif linha[0] == 'calc_modes_basis':
                    calc_modes_basis = true_or_false(linha[1], calc_modes_basis)
                elif linha[0] == 'calc_IBL_way':
                    calc_IBL_way = true_or_false(linha[1], calc_IBL_way)
                elif linha[0] == 'Calculate_Kernel':
                    Calculate_Kernel = true_or_false(linha[1], Calculate_Kernel)                    
                elif linha[0] == 'write_DKernel':
                    write_DKernel = true_or_false(linha[1], write_DKernel)
                elif linha[0] == 'just_RPA_diag':
                    just_RPA_diag = true_or_false(linha[1], just_RPA_diag)
                elif linha[0] == 'report_RPA_data':
                    report_RPA_data = true_or_false(linha[1], report_RPA_data)
                elif linha[0] == 'read_Akcv_trick':
                    read_Akcv_trick = true_or_false(linha[1], read_Akcv_trick)
                elif linha[0] == 'show_imag_part':
                    show_imag_part = true_or_false(linha[1], show_imag_part)
                elif linha[0] == 'acoutic_sum_rule':
                    acoutic_sum_rule = true_or_false(linha[1], acoutic_sum_rule)
                elif linha[0] == 'use_hermicity_F':
                    use_hermicity_F = true_or_false(linha[1], use_hermicity_F) 
                elif linha[0] == 'log_k_points':
                    log_k_points = true_or_false(linha[1], log_k_points)

########## did not recognize this line #############

                elif linha[0][0] != '#':
                    print('Parameters not recognized in the following line:\n')
                    print(line)
        arq_in.close()

    

read_input('forces.inp')
print('\n--------------------Configurations----------------------------\n\n')

print('Exciton index to be read : '+str(iexc))
print(f'Eqp data file : {eqp_file}')
print(f'Exciton file : {exciton_file}')
print(f'Elph directory : {el_ph_dir}')

print(f'Using Acoustic Sum Rule for elph coeffs : {acoutic_sum_rule}')
print(f'Using "hermicity" in forces calculations : {use_hermicity_F}')

if calc_IBL_way == True:
    print('Calculating derivatives of Kernel using Ismail-Beigi and Louie\'s paper approach')


print('\n-------------------------------------------------------------\n\n')