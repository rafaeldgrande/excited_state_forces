
ESF_DIR="/home/rrodriguesdelgrand/programs/excited_state_forces"  # directory where the excited_state_forces code is saved

python $ESF_DIR/elph/elph_xml_to_h5.py --elph_dir ../0-GWBSE/DFT/4-wfn_co/_ph0/LiF.phsave/ --wfn_co ../0-GWBSE/DFT/4-wfn_co/WFN.h5 --wfn_fi ../0-GWBSE/DFT/5-wfn_fi/WFN.h5 --dtmat ../0-GWBSE/GWBSE/4-absorption/dtmat &> elph_xml_to_h5.out
