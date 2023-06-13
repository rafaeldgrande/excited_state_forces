# Excited State Forces
Excited state forces code. Calculate forces after excitation by combining results from GW/BSE and DFPT calculations



Important notes (to be organized later)

1 - Use the same scf calculation as starting point for both gw/bse and DFPT workflows. If you use two different scf calculations (even with the same input file), it is possible that the eigencvecs from one calculation to other are different from each other by a phase factor or different signs.
