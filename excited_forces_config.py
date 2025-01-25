# IGNORE ERRORS
IGNORE_ERRORS = False


from scipy.constants import physical_constants

# File with variables used in excited_forces.py code
# It reads input forces.inp file
# If file not found or some parameter not find in file
# then the code uses (dumb) default values


# Conventions
TOL_DEG = 1e-5  # tolerance to see if two energy values are degenerate  ----------

# Conversion factors
Ry2eV = physical_constants["Rydberg constant times hc in eV"][0]
bohr2A = physical_constants["Bohr radius"][0]*1e10

# Ry2eV = 13.6056980659
# bohr2A = 0.529177249

# Dumb default parameters
iexc = 1
jexc = -1  # if it keeps to be jexc, then make jexc to iexc
factor_head = 1
ncbnds_sum = -1  # how many c/v bnds to be included in forces calculation?
nvbnds_sum = -1  # if == -1, then uses all used bnds in the BSE hamiltonian
# # files and paths to be opened
eqp_file = 'eqp1.dat'
exciton_file = 'eigenvectors.h5'
el_ph_dir = './'
kernel_file = 'bsemat.h5'

# conditionals

calc_modes_basis = False    # not being used yet

write_DKernel = False    # not being used  yet
Calculate_Kernel = False    # Dont change. We dont know how to work with kernel yet

just_RPA_diag = False    # If true doesn't calculate forces a la David
report_RPA_data = False    # report Fcvkc'v'k' matrix elements.

# Show imaginary part of excited state force . It should be very close to 0 when iexc = jexc
show_imag_part = False

# Makes the excited state forces to sum to zero (obey Newton's third law), by making sum of elph matrix elems to 0. May want to set this variable to true
acoutic_sum_rule = True
use_hermicity_F = True     # Use the fact that F_cvc'v' = conj(F_c'v'cv)
# Reduces the number of computed terms by about half

log_k_points = False     # Write k points used in BSE and DFPT calculations

# reads Acvk coeffs from files produced by my modified version of 
# summarize_eigenvectors.x
read_Acvk_pos = False
Acvk_directory = './' # directory where the Acvk files are

# do not renormalize elph coefficients (make <n|dHqp|m> = <n|dHdft|m> for all n and m)
no_renorm_elph = False
# print('!!!!!!!!!!', no_renorm_elph)

# make elph interpolation "a la BerkeleyGW code" using 
# the "dtmat_non_bin_val" and "dtmat_non_bin_conds" files
# those are output from the modified version of the absorption code
elph_fine_a_la_bgw = False

# parameters for interpolation 
# in the absorption.inp file, are the following
#  number_val_bands_coarse 10
#  number_val_bands_fine 5
#  number_cond_bands_coarse 8
#  number_cond_bands_fine 5

ncbands_co, nvbands_co = 0, 0
nkpnts_co = 0

# write dK (derivative of kernel) matrix elements
write_dK_mat = False

# do not check kpoints between bgw and dfpt calculations
# trust that both codes did the calculation in the same order
# so we do not need to map one grid in another
trust_kpoints_order = False

# is this calculation with spin triplet (True) or spin singlet (False)? K = Kd
spin_triplet = False

# is this calculation with local_fields flag? K = Kx
local_fields = False

# run in parallel flag
run_parallel = False

# modify Acvk to be Acvk = delta_(cvk,cvk)
# where cvk is the transition for which Acvk is originally maximum
use_Acvk_single_transition = False

# List of ELPH to be read. If list is empty, then all are read
dfpt_irreps_list = []

# If true limit sum of excited state forces to be done only on coefficients ik, ic, iv
# listed in file "indexes_limited_sum_BSE.dat". The file has the following format
# 1 1 1
# 1 1 2
# 1 1 3
# 1 2 1
# 1 2 2
# 1 2 3
# where the first collumn is the ik, the second is the ic, and the third is the iv
# this is useful when one know a priori which transitions are the most relevant
limit_BSE_sum = False

# Number between 0.0 and 1.0
# set the coefficients ic, iv and ik for which sum |A_cvk|^2 <= limit_BSE_sum_up_to_value
# if it is equal to 0, then all coefficients are used
# if it is different than 0, than make limit_BSE_sum = False, and ignore indexes_limited_sum_BSE.dat file
limit_BSE_sum_up_to_value = 1.0

# use vectorized sums
# F_{mu,k,c1,v1,c2,v2} = A_{kcv1}^* A_{kc2v2} (g_{mu,k,c1,c2}*delta(v1,v2) - g_{mu,k,v1,v2}*delta(c1,c2))
# F_{mu} = sum_{k,c1,v1,c2,v2} F_{mu,k,c1,v1,c2,v2}
# Create the matrices A_{kcv1}^* A_{kc2v2} with shape nk, nc, nc, nv, nv
# g_{mu,k,c1,c2}*delta(v1,v2) with shape nk, nc, nc, nv, nv
# and g_{mu,k,v1,v2}*delta(c1,c2) with shape nk, nc, nc, nv, nv
do_vectorized_sums = True

def true_or_false(text, default_value):
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False
    else:
        return default_value



def read_input(input_file):

    # getting necessary info

    global iexc, jexc, factor_head
    global nvbnds_sum, ncbnds_sum
    global eqp_file, exciton_file, el_ph_dir
    global dyn_file, kernel_file
    global calc_modes_basis
    global write_DKernel, Calculate_Kernel
    global just_RPA_diag, report_RPA_data
    global show_imag_part
    global read_Acvk_pos, Acvk_directory
    global acoutic_sum_rule, use_hermicity_F
    global log_k_points
    global no_renorm_elph   
    global elph_fine_a_la_bgw                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    global nkpnts_co, nvbands_co, ncbands_co
    global write_dKE_mat, write_dK_mat
    global trust_kpoints_order
    global spin_triplet
    global local_fields
    global run_parallel
    global use_Acvk_single_transition
    global dfpt_irreps_list
    global limit_BSE_sum, limit_BSE_sum_up_to_value
    global do_vectorized_sums

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
                elif linha[0] == 'jexc':
                    jexc = int(linha[1])
                elif linha[0] == 'ncbnds_sum':
                    ncbnds_sum = int(linha[1])
                elif linha[0] == 'nvbnds_sum':
                    nvbnds_sum = int(linha[1])
                elif linha[0] == 'factor_head':
                    factor_head = float(linha[1])
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
                elif linha[0] == 'Acvk_directory':
                    Acvk_directory = linha[1]
                elif linha[0] == 'ncbands_co':
                    ncbands_co = int(linha[1])
                elif linha[0] == 'nvbands_co':
                    nvbands_co = int(linha[1])
                elif linha[0] == 'nkpnts_co':
                    nkpnts_co = int(linha[1])
                elif linha[0] == 'limit_BSE_sum_up_to_value':
                    limit_BSE_sum_up_to_value = float(linha[1])
                    
                ######### conditionals #############################

                elif linha[0] == 'calc_modes_basis':
                    calc_modes_basis = true_or_false(
                        linha[1], calc_modes_basis)
                elif linha[0] == 'Calculate_Kernel':
                    Calculate_Kernel = true_or_false(
                        linha[1], Calculate_Kernel)
                elif linha[0] == 'write_DKernel':
                    write_DKernel = true_or_false(linha[1], write_DKernel)
                elif linha[0] == 'just_RPA_diag':
                    just_RPA_diag = true_or_false(linha[1], just_RPA_diag)
                elif linha[0] == 'report_RPA_data':
                    report_RPA_data = true_or_false(linha[1], report_RPA_data)
                elif linha[0] == 'show_imag_part':
                    show_imag_part = true_or_false(linha[1], show_imag_part)
                elif linha[0] == 'acoutic_sum_rule':
                    acoutic_sum_rule = true_or_false(linha[1], acoutic_sum_rule)
                elif linha[0] == 'use_hermicity_F':
                    use_hermicity_F = true_or_false(linha[1], use_hermicity_F)
                elif linha[0] == 'log_k_points':
                    log_k_points = true_or_false(linha[1], log_k_points)
                elif linha[0] == 'read_Acvk_pos':
                    read_Acvk_pos = true_or_false(linha[1], read_Acvk_pos)
                elif linha[0] == 'no_renorm_elph':
                    no_renorm_elph = true_or_false(linha[1], no_renorm_elph)
                elif linha[0] == 'elph_fine_a_la_bgw':
                    elph_fine_a_la_bgw = true_or_false(linha[1], elph_fine_a_la_bgw)
                elif linha[0] == 'write_dK_mat':
                    write_dK_mat = true_or_false(linha[1], write_dK_mat)
                elif linha[0] == 'trust_kpoints_order':
                    trust_kpoints_order = true_or_false(linha[1], trust_kpoints_order)
                elif linha[0] == 'spin_triplet':
                    spin_triplet = true_or_false(linha[1], spin_triplet)
                elif linha[0] == 'local_fields':
                    local_fields = true_or_false(linha[1], local_fields)
                elif linha[0] == 'run_parallel':
                    run_parallel = true_or_false(linha[1], run_parallel)
                elif linha[0] == 'use_Acvk_single_transition':
                    use_Acvk_single_transition = true_or_false(linha[1], use_Acvk_single_transition)
                elif linha[0] == 'dfpt_irreps_list':
                    for i in range(1, len(linha)):
                        dfpt_irreps_list.append(int(linha[i])-1)
                elif linha[0] == 'limit_BSE_sum':
                    limit_BSE_sum = true_or_false(linha[1], limit_BSE_sum)
                elif linha[0] == 'do_vectorized_sums':
                    do_vectorized_sums = true_or_false(linha[1], do_vectorized_sums)

                
########## did not recognize this line #############

                elif linha[0][0] != '#':
                    print('Parameters not recognized in the following line:')
                    print(line)
        arq_in.close()

    if jexc == -1:
        jexc = iexc


read_input('forces.inp')

print('\n--------------------Configurations----------------------------\n\n')


print(f'Eqp data file : {eqp_file}')
print(f'Exciton file : {exciton_file}')
print(f'Elph directory : {el_ph_dir}')

print(f'Using Acoustic Sum Rule for elph coeffs : {acoutic_sum_rule}')

if jexc == iexc:
    print(f'Exciton index to be read : {iexc}')
else:
    print(f'Exciton indexes to be read : {iexc}, {jexc}')
    print(f'As excitons indexes are different, we must use complex values to forces!')
    print(f'Setting show_imag_part to true')
    show_imag_part = True

if iexc == jexc:
    print(f'Using "hermicity" in forces calculations : {use_hermicity_F}')
elif use_hermicity_F == True:
    print(f'Exciton indexes are not equal to each other. Cannot use "hermicity"')
    print(f'Setting use_hermicity_F to false')
    use_hermicity_F = False

if no_renorm_elph == True:
    print('Elph coefficients at gw level will be considered to be equal to coefficients calculated at DFT level')
    
if local_fields == True and spin_triplet == True:
    print('Warning! Both spin_triplet and local_fields are true! Choose just one!')
    print('Making local_fields = False and spin_triplet = False')
    local_fields = False
    spin_triplet = False
    
if limit_BSE_sum_up_to_value < 1.0:
    if limit_BSE_sum == True:
        print('Warning! limit_BSE_sum_up_to_value < 1.0 and limit_BSE_sum = True. Setting limit_BSE_sum = False')
        limit_BSE_sum = False

print('\n-------------------------------------------------------------\n\n')
