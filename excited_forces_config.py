
# File with variables used in excited_forces.py code
# It reads input forces.inp file
# If file not found or some parameter not find in file
# then the code uses (dumb) default values



TOL_DEG = 1e-5  # tolerance to see if two energy values are degenerate
Ry2eV = 13.6056980659
bohr2A = 0.529177249

# Dumb default parameters
Nkpoints = 1
Nvbnds = 1
Ncbnds = 1
Nval = 1
Nat = 1
iexc = 1
alat = 10 # FIXME: read it from input file or other source (maybe read volume instead)
# files and paths to be opened 
eqp_file = 'eqp1.dat'
exciton_dir = './'
el_ph_dir = './'
kernel_file = 'bsemat.h5'

# conditionals
just_real = False
calc_modes_basis = False
calc_IBL_way = True
write_DKernel = False
report_RPA_data = False
just_RPA_diag = False
Calculate_Kernel = False
read_Akcv_trick = True

def read_input(input_file):

    # getting necessary info

    global alat
    global Nkpoints, Nvbnds, Ncbnds, Nval
    global Nat, iexc
    global eqp_file, exciton_dir, el_ph_dir
    global dyn_file, kernel_file
    global just_real, calc_modes_basis
    global calc_IBL_way

    try:
        arq_in = open(input_file)
        print(f'Reading input file {arq_in}')
    except:
        print(f'File {arq_in} not found')

    for line in arq_in:
        linha = line.split()
        if len(linha) >= 2:
            if linha[0] == 'Nkpoints':
                Nkpoints = int(linha[1])
            elif linha[0] == 'Nvbnds':
                Nvbnds = int(linha[1])
            elif linha[0] == 'Ncbnds':
                Ncbnds = int(linha[1])
            elif linha[0] == 'Nval':
                Nval = int(linha[1])
            elif linha[0] == 'Nat':
                Nat = int(linha[1])
            elif linha[0] == 'iexc':
                iexc = int(linha[1])
            elif linha[0] == 'eqp_file':
                eqp_file = linha[1]
            elif linha[0] == 'exciton_dir':
                exciton_dir = linha[1]
            elif linha[0] == 'el_ph_dir':
                el_ph_dir = linha[1]
            elif linha[0] == 'just_real':
                if linha[1] == 'True':
                    just_real = True
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
            elif linha[0] == 'alat':
                alat = float(linha[1])
            elif linha[0][0] != '#':
                print('Parameters not recognized in the following line:\n')
                print(line)

    arq_in.close()

read_input('forces.inp')
