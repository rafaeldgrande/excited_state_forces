
import numpy as np
import sys

file_scf_input = sys.argv[1]
file_forces = sys.argv[2]
flavor_forces = int(sys.argv[3])
output_file = sys.argv[4]

def get_atoms_from_QE_file_0(file_scf_input):
    
    arq = open(file_scf_input, "r")
    
    ATOMS = []
    CELL_LATT = []
    
    for line in arq:
        line_split = line.split()
        if len(line_split) > 0:
            
            if line_split[0] == "nat":
                Natoms = int(line_split[2])
                
            if line_split[0] == "CELL_PARAMETERS":
                for i in range(3):
                    line_split = arq.readline().split()
                    CELL_LATT.append(f"""{line_split[0]}   {line_split[1]}   {line_split[2]}""")
            
            if line_split[0] == "ATOMIC_POSITIONS":
                for iatom in range(Natoms):
                    line_split = arq.readline().split()
                    ATOMS.append(f"""{line_split[0]}   {line_split[1]}   {line_split[2]}   {line_split[3]}    """)
    
    arq.close()
    return ATOMS, CELL_LATT

def read_excited_forces(excited_state_forces_file, flavor_forces):
    # flavor = 1 -> RPA_diag 
    # flavor = 2 -> RPA_diag_offiag 
    
    data = np.loadtxt(excited_state_forces_file, usecols=flavor_forces+1)
    data = data.reshape(-1, 3)
    return data

forces = read_excited_forces(file_forces, flavor_forces)

ATOMS, CELL_LATT = get_atoms_from_QE_file_0(file_scf_input)
Nat = len(ATOMS)

arq_out = open(output_file, "w")

arq_out.write("""ANIMSTEPS  1
CRYSTAL 
PRIMVEC \n""")

for ilat in range(3):
    arq_out.write(f"""    {CELL_LATT[ilat]}\n""")
    
arq_out.write(f"""PRIMCOORD    1
{Nat}   1\n""")

for iatom in range(Nat):
    fx, fy, fz = forces[iatom]
    arq_out.write(f"""    {ATOMS[iatom]}  {fx}   {fy}    {fz}\n""")
    
arq_out.close()
