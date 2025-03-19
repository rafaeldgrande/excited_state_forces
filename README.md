# Excited State Forces

Excited state forces code. Calculate forces after excitation by combining results from GW/BSE and DFPT calculations. 

Details on the implementation and benchmarks can be found here: https://arxiv.org/abs/2502.05144 

The excited force expression is given by:

$$ F = \sum_{\nu cvc'v' k} A_{cvk} A^*_{c'v'k} (g^{\nu}_{cc'k} - g^{\nu}_{vv'k}) \hat{\nu} $$
$$ \vec{F} = \sum_{\nu cv c'v' k} $$

where \(\hat{\nu}\) is one displacement pattern (a phonon mode for example), \(A_{cvk}\) is the exciton coefficient obtained from the Bethe-Salpeter Equation, and \(g^{\nu}_{ijk}\) is the electron-phonon coefficient \(\langle ik | \partial_{\nu} V | ij \rangle\).

If you are using our code, please cite 

```
@misc{delgrande2025revisitingabinitioexcitedstate,
      title={Revisiting ab-initio excited state forces from many-body Green's function formalism: approximations and benchmark}, 
      author={Rafael R. Del Grande and David A. Strubbe},
      year={2025},
      eprint={2502.05144},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2502.05144}, 
}
```


