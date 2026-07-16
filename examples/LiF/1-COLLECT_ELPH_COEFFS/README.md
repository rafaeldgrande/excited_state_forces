
Here we are going to collect the elph coefficients from the DFPT calculation.
Quantum Espresso saves those elph coefficients in the _ph0/LiF.phsave/ directory

QE computes the matrix element <n,k+q|dV_nu(q)|m,k>, where nu is a displacement pattern
When there is no symmetry the displacement patterns are cartesian displacement of atoms,
and when quantum espresso captures the symmetry of system it guesses what will be the phonon
modes and use those displacements. Those displacement patterns are written in the patterns.1.xml
file. Elph coefficients are written in the elph.1.<mu>.xml files.

Notice that here we are using only q=0, because this is what is necessary for excited state forces calculations

What the $ESF_DIR/elph/elph_xml_to_h5.py code does here is:

1. Reads .xml patterns and elph files in the coarse grid and converts it cartesian displacements
2. Reads WFN_co.h5, hdf5 file for the wavefunctions in the coarse grid where the elph coefficients were calculated
It reads informaiton like number of valence bands, atomic positions, etc
3. Apply acoustic sum rule on the elph coefficients by imposing sum_atoms <nk+q|dV_alpha(q)|mk> = 0, with alpha = x,y,z
We found that this was the reason why the forces on C and O atoms were different in PRL 90, 076401 (2003).
4. Filter elph coefficients into valence and conduction blocks (to be used by main/excited_forces.py)
5. Save elph coefficients in a hdf5 file (default name: elph_orig_kgrid.h5)
hdf5 files are faster to be read and manipulated than .xml so one can explore the elph coefficients
6. Interpolates elph coefficients from a coarse to a fine grid using BerkeleyGW interpolation scheme by expanding 
wavefunctions in the coarse into wavefunctions of the fine grid (computing <n,kfi|m,kco>. This is enconded in the 
binary file dtmat produced in the absorption step. WFN_fi.h5 is the wavefunction file used as a fine grid in the 
absorption step of BSE calculations. 

To run the code do (don't forget to change ESF_DIR variable to the path where you save the code)
bash get_elph.bash

it runs python python $ESF_DIR/elph/elph_xml_to_h5.py --elph_dir ../0-GWBSE/DFT/4-wfn_co/_ph0/LiF.phsave/ --wfn_origin ../0-GWBSE/DFT/4-wfn_co/WFN.h5 --wfn_to_interpolate ../0-GWBSE/DFT/5-wfn_fi/WFN.h5 --dtmat ../0-GWBSE/GWBSE/4-absorption/dtmat &> elph_xml_to_h5.out 

$ESF_DIR/elph/elph_xml_to_h5.py -h prints a help message showing possible flags

Note: To convert binary to hdf5 use wfn2hdf.x BIN wfn.complex WFN.h5, where wfn2hdf.x is part of the BerkeleyGW package
