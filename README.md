# Excited State Forces

Excited state forces code. Calculate forces after excitation by combining results from GW/BSE and DFPT calculations.

The excited force expression is given by:

\[ 
F = \sum_{\nu cvc'v' k} A_{cvk} A^*_{c'v'k} (g^{\nu}_{cc'k} - g^{\nu}_{vv'k}) \hat{\nu} 
\]

where \(\hat{\nu}\) is one displacement pattern (a phonon mode for example), \(A_{cvk}\) is the exciton coefficient obtained from the Bethe-Salpeter Equation, and \(g^{\nu}_{ijk}\) is the electron-phonon coefficient \(\langle ik | \partial_{\nu} V | ij \rangle\).
