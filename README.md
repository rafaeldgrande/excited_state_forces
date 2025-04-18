# Excited State Forces

Excited state forces code. Calculate forces after excitation by combining results from GW/BSE and DFPT calculations. 

Details on the implementation and benchmarks can be found here: https://arxiv.org/abs/2502.05144 

The excited force expression is given by:

$$ \vec{F} = \sum_{\mu k cv c'v'} \hat{\mu} A_{kcv} A_{kc'v'} \left( g^{\mu}_ {kc,kc'} \delta(v,v') - g^{\mu}_{kv,kv'} \delta(c,c') \right) $$

where $\hat{\mu}$ is one displacement pattern (a phonon mode for example), $A_{cvk}$ is the exciton coefficient obtained from the Bethe-Salpeter Equation, and $g^{\nu}_ {ki,kj}$ is the electron-phonon coefficient connecting bands $i$ and $j$ at k point $k$.

If you are using our code, please cite 

```
@misc{delgrande2025,
      title={Revisiting ab-initio excited state forces from many-body Green's function formalism: approximations and benchmark}, 
      author={Rafael R. Del Grande and David A. Strubbe},
      year={2025},
      eprint={2502.05144},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2502.05144}, 
}
```


