# Excited State Forces

## Overview

This code calculates excited state forces after electronic excitation using a many-body Green's function formalism. The implementation combines exciton coefficients from the Bethe-Salpeter Equation (BSE) with electron-phonon coupling coefficients from DFPT to compute forces in excited states.


## Theory

The excited force expression is given by:

$$ \vec{F} = \nabla \Omega = \sum_{\mu k cv c'v'} \hat{\mu} A_{kcv} A_{kc'v'} \left( g^{\mu}_ {kc,kc'} \delta(v,v') - g^{\mu}_{kv,kv'} \delta(c,c') \right) $$

where:
- $\hat{\mu}$ is a displacement pattern (phonon mode)
- $A_{cvk}$ are exciton coefficients from the Bethe-Salpeter Equation
- $g^{\nu}_{ki,kj}$ are electron-phonon coefficients connecting bands $i$ and $j$ at k-point $k$
- $c,v$ represent conduction and valence band indices
- $k$ represents k-point indices

**For detailed implementation and benchmarks, see:** https://arxiv.org/abs/2502.05144


### External Software Dependencies

- **BerkeleyGW**: For GW/BSE calculations
- **Quantum ESPRESSO**: For DFT ground state and DFPT calculations


## Usage

### Basic Usage

1. **Prepare input files**: You need results from both GW/BSE and DFPT calculations
   - `eigenvectors.h5`: Exciton eigenvectors from BSE calculation
   - `eqp.dat`: Quasiparticle energies from GW calculation  
   - `*.phsave/`: Directory containing electron-phonon coupling data from DFPT

2. **Create a forces.inp file**:
```
iexc 1
eqp_file         eqp.dat
exciton_file     eigenvectors.h5
el_ph_dir        system.phsave/
```

3. **Run the calculation**:
```bash
python excited_forces.py
```

### Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `iexc` | Excited state index to calculate forces for | 1 |
| `jexc` | Second exciton index (for cross terms) | same as iexc |
| `factor_head` | Scaling factor for head of dielectric matrix | 1 |
| `ncbnds_sum` | Number of conduction bands in sum | all available |
| `nvbnds_sum` | Number of valence bands in sum | all available |
| `eqp_file` | Quasiparticle energies file | eqp.dat |
| `exciton_file` | BSE eigenvectors file | eigenvectors.h5 |
| `el_ph_dir` | Electron-phonon coupling directory | *.phsave/ |

## Examples

The repository includes complete examples for two systems:

### CO (Carbon Monoxide)
Located in `examples/CO/`, this example demonstrates:
- Small molecule excited state forces
- Complete workflow from DFT to excited state forces

### LiF (Lithium Fluoride) 
Located in `examples/LiF/`, this example shows:
- Ionic crystal excited state forces
- Workflow for extended systems

Each example includes:
- Input files for all calculation steps
- Job submission scripts
- Linking scripts for file management
- Complete documentation of the workflow

### Running Examples

```bash
cd examples/CO/
# Follow the numbered directories 1-8 for the complete workflow
cd 8-excited_state_forces/
python ../../../excited_forces.py
```


## Module Structure

- `excited_forces.py`: Main execution script
- `excited_forces_m.py`: Core force calculation functions
- `excited_forces_classes.py`: Data structure classes
- `excited_forces_config.py`: Configuration file parser
- `bgw_interface_m.py`: BerkeleyGW file interface
- `qe_interface_m.py`: Quantum ESPRESSO interface
- `visualize_forces.py`: Force visualization utilities

## Citation

If you use this code in your research, please cite:

```bibtex
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




