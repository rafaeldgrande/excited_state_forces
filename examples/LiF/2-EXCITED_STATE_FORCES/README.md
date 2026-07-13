
To execute the excited state forces code 
bash exc_forces.bash

This is the minimum example of running the ESF code. 
It needs the elph coefficients (elph_fine.h5), quasiparticle energy levels (eqp.dat) and exciton coefficients (eigenvectors.h5).
iexc tells the exciton index to which the ESF will be computed.  

The output exc_forces_1_1_cart.dat is exc_forces_1_1_cart.dat which are the excited state forces in cartesian basis.

Exercise:
Change iexc to other values and compute excited state forces for other excitons

The gradient of the excitation energy for exciton iexc is given by 

d Omega_iexc / dr = <iexc | dH/dr | iexc>

in other words a diagonal exciton-phonon matrix element. You can also compute off-diagonal (in general complex numbers) matrix elements by using 

iexc 1
jexc 2

which will compute <1|dH/dr|2> and save the file exc_forces_1_2_cart.dat

