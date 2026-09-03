rec_cm_to_eV = 1.239841984e-4    # cm^-1 to eV
k_B          = 8.617333262145e-5  # Boltzmann constant in eV/K
hbar         = 6.582119569e-16   # reduced Planck constant in eV*s
ry2eV        = 13.605693122994   # Rydberg to eV
Ry2eV        = ry2eV             # alias (capital-R variant used in main/)
eV2ry        = 1.0 / ry2eV
bohr2A       = 0.529177210903    # Bohr radius in Angstrom
TOL_ZERO     = 1e-6              # degeneracy / near-zero threshold

# Force-constant -> (hbar*omega)^2 conversion, for mass-weighted dynamical-matrix
# elements built as D_nunu' = sum u^nu_s C_ss' u^nu'_s' with eigenDISPLACEMENTS
# u^nu_s = e^nu_s/sqrt(M_kappa) (M in amu, C in eV/Ang^2, e the Euclidean-unit-
# normalized real-space displacement pattern parse_matdyn_modes returns):
#   (hbar*omega_nu)^2 [eV^2] = FC_TO_HW2 * D_nunu [eV/(Ang^2 * amu)]
# Derived from hbar, amu, eV, Angstrom SI values (scipy.constants), verified
# 2026-08-19 against a known mode (411.6 cm^-1, pure-S effective mass 32.06 amu)
# giving a physically sensible ~20 eV/Ang^2 diagonal force constant. See
# exciton_phonon_renorm/io_data.py's module docstring for the full derivation
# and why this is needed (the exciton-phonon coupling data in this project's
# forces/ph/* files is projected with the *un*-mass-weighted eigenvector e,
# not the eigendisplacement u -- PLAN.md's own "classic trap").
FC_TO_HW2 = 4.180159279778997e-3  # eV^2 per (eV/Ang^2/amu)
