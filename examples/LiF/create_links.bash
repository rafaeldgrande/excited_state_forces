
PREFIX='LiF'

cd 5-epsilon/
ln -sf ../2-wfn/wfn.complex WFN
ln -sf ../3-wfnq/wfn.complex WFNq
cd ../

cd 6-sigma/
ln -sf ../2-wfn/rho.complex RHO
ln -sf ../4-wfn_co/vxc.dat .
ln -sf ../2-wfn/wfn.complex WFN_inner
ln -sf ../4-wfn_co/wfn.complex WFN_outer
ln -sf ../5-epsilon/eps0mat.h5 . 
ln -sf ../5-epsilon/epsmat.h5 .
cd ../

cd 7-kernel/
ln -sf ../5-epsilon/eps0mat.h5 .
ln -sf ../5-epsilon/epsmat.h5 .
ln -sf ../4-wfn_co/wfn.complex WFN_co
cd ../

cd 8-absorption/
ln -sf ../5-epsilon/eps0mat.h5 .
ln -sf ../5-epsilon/epsmat.h5 .
ln -sf ../4-wfn_co/wfn.complex WFN_co
ln -sf ../1-scf_fi/wfn.complex WFN_fi
ln -sf ../6-sigma/eqp1.dat eqp_co.dat
ln -sf ../7-kernel/bsemat.h5 .
cd ../
