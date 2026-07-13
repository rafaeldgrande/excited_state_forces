

Results from this example directory will be HIGLY UNCONVERGED. 
To perform converged production calculations one needs to converge 
DFT, DFPT and GW/BSE calculations with Quantum Espresso and BerkeleyGW
Check those codes tutorials to be confident within this workflow.

Steps:

1. Create symbolic links
bash create_links.bash

This will create symbolic links for different steps of the GWBSE workflow

2. Run DFT, DFPT  
sbatch job_DFT.sub

Adapt this for your cluster. This submission file executes the following steps

Quantum Espresso calculations

2.a In 1-scf_fi it performs scf calculations. From here we need the charge density
file to be used in the next nscf calculations. 

2.b 2-wfn we compute bands calculations that generates a large number of bands conduction 
states to be used to compute the dielectric function and self-energy operator

2.c 3-wfnq we compute bands calculations with small q shift to be used to compute 
the dielectric function and self-energy operator

2.d 4-wfn_co bands to be used as basis for GW calculations (WFN_outer) and as 
WFN_co in the BSE calculations

2.e 5-wfn_fi bands to be used as a fine grid in the BSE. Last we perform DFPT calculations
with electron_phonon='simple', which tells ph.x to compute elph coefficients.
Here we are computing the ELPH directly in the fine grid, but it could be done in the coarse
grid, and then interpolated to a fine grid (to be explained later!).
Notice that we added a smearing in the scf calculations, even though
we dealing here with a semiconductor. This is necessary to "trick" the ph.x
that is projected to compute elph just for metallic systems.

3. Run GWBSE:
sbatch job_GWBSE.sub

3.a 5-epsilon calculation of dielectric function

3.b 6-sigma GW energy level calculations

3.c 7-kernel Calculations of electron-hole interactions in the coarse grid

3.d 8-absorption Interpolation of GW energy levels and kernel from the coarse 
to the fine grids, builds the BSE Hamiltonian in the fine grid, 
diagonilize it and then it computes the optical absorption.
Here we have the following adaptations from a typical gwbse workflow:
- WFN_fi the wavefunction in the fine grid is the same dir where we run DFPT. The reason for that
is that the wavefunctions used to build the BSE Hamiltonian need to be the same as the 
ones used to calculate the elph coefficients. If they are different then the complex phases
between the electron-phonon and exciton coefficients will not be consistent
- In the absorption script we use the flag use_momentum instead of use_velocity (see the end of 
section 3 of Comput. Phys. Commun. Vol 183, Issue 6, Pages 1269-1289 (2012)).
The velocity operator gives the transition dipole moment matrix elements including non local 
effects but it needs two fine grids, namely WFN_fi and WFNq_fi, which means one would need to 
perform two DFPT calculations in those two grids, which is not implemented in our workflow.
With use_momentum the transition dipole moments do not include non-local effects and the BSE
Hamiltonian is just build with one WFN_fi file. There is no correlation between the intensity
of excited state forces and the intensity of transition dipole moments of excitons. If one wants 
to calculate both the excited state forces and an accurate absorption spectra we recomend
to create two directories 8-absorption_vel and 8-absorption-mom, where one runs absorption.flavor.x
using use_velocity and use_momentum, respectively, as the absorption step usually does not demand
a lot of computational resources.

