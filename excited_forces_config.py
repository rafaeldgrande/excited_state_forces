
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
calc_IBL_way = True         # I should comment latter
write_DKernel = False       # not being used 
report_RPA_data = False      # report Fcvkc'v'k' matrix elements.
just_RPA_diag = False       # If true doesn't calculate forces a la David
Calculate_Kernel = False    # Dont change. We dont know how to work with kernel yet
read_Akcv_trick = False     # If false, read eigenvectors.h5 file, when BGW is compiled with hdf5. If true read Acvk{iexc} that is an output of my modified summarize_eigenvectors.f90 file
show_imag_part = False      # Show imaginary part of excited state force . It should be very close to 0 
use_F_complex_conj = False  # In development. don't change
acoutic_sum_rule = True     # Makes the excited state forces to sum to zero (obey Newton's third law), by making sum of elph matrix elems to 0. May want to set this variable to true
use_hermicity_F = True # Use the fact that F_cvc'v' = conj(F_c'v'cv)
                       # Reduces the number of computed terms by about half
log_k_points = True          # Write k points used in BSE and DFPT calculations

def read_input(input_file):

    # getting necessary info

    global alat
    # global Nkpoints, Nvbnds, Ncbnds, Nval
    # global Nat
    global iexc
    global eqp_file, exciton_file, el_ph_dir
    global dyn_file, kernel_file
    global calc_modes_basis
    global calc_IBL_way
    global show_imag_part

    try:
        arq_in = open(input_file)
        print(f'Reading input file {input_file}')

        for line in arq_in:
            linha = line.split()
            if len(linha) >= 2:
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
                elif linha[0] == 'calc_modes_basis':
                    if linha[1] == 'True':
                        calc_modes_basis = True
                elif linha[0] == 'calc_IBL_way':
                    if linha[1] == 'True':
                        calc_IBL_way = True
                elif linha[0] == 'show_imag_part':
                    show_imag_part = float(linha[1])                    
                elif linha[0][0] != '#':
                    print('Parameters not recognized in the following line:\n')
                    print(line)
        arq_in.close()
    except:
        print(f'File {input_file} not found')

    

read_input('forces.inp')


print('\n---- Parameters -----\n')
print('Exciton index to be read : '+str(iexc))
if calc_IBL_way == True:
    print('Calculating derivatives of Kernel using Ismail-Beigi and Louie\'s paper approach')
print('\n---------------------\n\n')