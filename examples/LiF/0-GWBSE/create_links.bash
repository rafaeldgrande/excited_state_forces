
PREFIX='LiF'

cd GWBSE/

cd 1-epsilon/
ln -sf ../../DFT/2-wfn/wfn.complex WFN
ln -sf ../../DFT/3-wfnq/wfn.complex WFNq
cd ../

cd 2-sigma/
ln -sf ../../DFT/2-wfn/rho.complex RHO
ln -sf ../../DFT/4-wfn_co/vxc.dat .
ln -sf ../../DFT/2-wfn/wfn.complex WFN_inner
ln -sf ../../DFT/4-wfn_co/wfn.complex WFN_outer
ln -sf ../1-epsilon/eps0mat.h5 . 
ln -sf ../1-epsilon/epsmat.h5 .
cd ../

cd 3-kernel/
ln -sf ../1-epsilon/eps0mat.h5 .
ln -sf ../1-epsilon/epsmat.h5 .
ln -sf ../../DFT/4-wfn_co/wfn.complex WFN_co
cd ../

cd 4-absorption/
ln -sf ../1-epsilon/eps0mat.h5 .
ln -sf ../1-epsilon/epsmat.h5 .
ln -sf ../../DFT/4-wfn_co/wfn.complex WFN_co
ln -sf ../../DFT/5-wfn_fi/wfn.complex WFN_fi
ln -sf ../2-sigma/eqp1.dat eqp_co.dat
ln -sf ../3-kernel/bsemat.h5 .
cd ../
