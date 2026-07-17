# Excited State Forces - old version

This version is the one used to make most benchmarks and tests in https://arxiv.org/abs/2502.05144 

Excited state forces code. Calculate forces after excitation by combining results from GW/BSE and DFPT calculations


Important notes

1 - Use the same scf calculation as starting point for both gw/bse and DFPT workflows. If you use two different scf calculations (even with the same input file), it is possible that the eigencvecs from one calculation to other are different from each other by a phase factor or different signs.
