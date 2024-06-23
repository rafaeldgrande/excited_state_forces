
PREFIX='CO'

cd 2-wfn_gw  
mkdir -p $PREFIX.save
cd $PREFIX.save/
ln -sf ../../1-scf/$PREFIX.save/data-file-schema.xml .
ln -sf ../../1-scf/$PREFIX.save/charge-density.hdf5 . 
cd ../../

cd 3-wfn_bse
mkdir -p $PREFIX.save
cd $PREFIX.save/
ln -sf ../../1-scf/$PREFIX.save/data-file-schema.xml .
ln -sf ../../1-scf/$PREFIX.save/charge-density.hdf5 .
cd ../../

cd 4-epsilon/
ln -sf ../2-wfn_gw/wfn.complex WFN
cd ../

cd 5-sigma/
ln -sf ../2-wfn_gw/rho.complex RHO
ln -sf ../2-wfn_gw/vxc.dat .
ln -sf ../2-wfn_gw/wfn.complex WFN_inner
ln -sf ../2-wfn_gw/wfn.complex WFN_outer
ln -sf ../4-epsilon/eps0mat.h5 . 
cd ../

cd 6-kernel/
ln -sf ../4-epsilon/eps0mat.h5 .
ln -sf ../3-wfn_bse/wfn.complex WFN_co
cd ../

cd 7-absorption/
ln -sf ../4-epsilon/eps0mat.h5 .
ln -sf ../3-wfn_bse/wfn.complex WFN_co
ln -sf ../3-wfn_bse/wfn.complex WFN_fi
ln -sf ../5-sigma/eqp1.dat eqp_co.dat
ln -sf ../6-kernel/bsemat.h5 .
cd ../
