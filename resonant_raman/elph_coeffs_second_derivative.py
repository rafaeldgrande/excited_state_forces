

import numpy as np

TOL_ZERO = 1e-6  # threshold below which states are considered degenerate
ry2eV = 13.605693122994  # conversion factor from Rydberg to eV
eV2ry = 1.0 / ry2eV

def compute_second_order_elph(elph, E, Displacements):
    """
    Compute second-order electron-phonon coupling coefficients via second-order
    perturbation theory:

        g_beta_knm^(2) = - sum_{l} g_beta_knl * g_beta_klm / (E_kn - E_kl)
                         - sum_{l} g_beta_knl * g_beta_klm / (E_km - E_kl)

    The sum over alpha runs over all cartesian (atom-direction) degrees of freedom.
    If `elph` is in the normal-mode basis it is rotated to cartesian first via
    `Displacements`.  The returned array is in the same cartesian basis (first
    index = displacement-pattern / atom-direction index).
    
    g_beta_knl has units of ry/bohr, while E_nk has unit of eV, so E_nk is converted to ry when loaded
    The units of g2 will be ry/bohr^2.

    Parameters
    ----------
    elph : np.ndarray, shape (nmodes, nkpoints, nbands, nbands)
        Electron-phonon coupling coefficients
    E : np.ndarray, shape (nbands, nkpoints)
        Single-particle energies.  E[n, k] = energy of band n at k-point k.
    Displacements : np.ndarray, shape (nmodes, Natoms, 3)
        Displacement patterns.  Displacements[mu, :] = [[x1,y1,z1], [x2,y2,z2], ...]

    Returns
    -------
    second_der_elph : np.ndarray, shape (nmodes, nkpoints, nbands, nbands)
        Second-order e-ph coefficients indexed by cartesian displacement (alpha).
    """
    
    # --- rotate elph to cartesian (atom-direction) basis -------------------
    # g_cart[alpha, k, n, m] = sum_modes  dot(Displacements[modes], unit_vector_for_alpha) * elph[modes, k, n, m]
    
    # elph has shape (nmodes, nk, nbands, nbands)
    # displacement shape (Nmodes, Natoms, 3)
    g_cart = np.zeros_like(elph)
    for imode in range(Displacements.shape[0]):
        disp_flattened = Displacements[imode].flatten()  # shape (3*Natoms,)
        for ialpha in range(disp_flattened.shape[0]):
            unit_vector = np.zeros_like(disp_flattened)
            unit_vector[ialpha] = 1.0
            g_cart[ialpha] += np.dot(disp_flattened, unit_vector) * elph[imode]


    Nmodes, Nk, nbands, _ = elph.shape
    term1 = np.zeros_like(g_cart)
    term2 = np.zeros_like(g_cart)
    
    for beta in range(Nmodes):
        for ik in range(Nk):
            for n in range(nbands):
                for m in range(nbands):
                    sum1 = 0.0
                    sum2 = 0.0
                    for l in range(nbands):
                        dE_nl = E[n, ik] - E[l, ik]
                        dE_ml = E[m, ik] - E[l, ik]
                        if np.abs(dE_nl) > TOL_ZERO:
                            sum1 += g_cart[beta, ik, n, l] * g_cart[beta, ik, l, m] / dE_nl
                        if np.abs(dE_ml) > TOL_ZERO:
                            sum2 += g_cart[beta, ik, n, l] * g_cart[beta, ik, l, m] / dE_ml
                    term1[beta, ik, n, m] = -sum1
                    term2[beta, ik, n, m] = -sum2

    return term1 + term2

def read_eqp_dat_file(eqp_file):
    bands_dft, bands_qp = [], []
    data = np.loadtxt(eqp_file)
    Nbnds = int(data[0, 3])
    band_indexes = data[1:Nbnds+1, 1]
    Kpoints = data[0::Nbnds+1][:, :3]
    Nk = len(Kpoints)
    print(f'Number of kpoints {Nk}')
    for ibnd in range(Nbnds):
        temp = data[ibnd+1::Nbnds+1]
        bands_dft.append(temp[:, 2])
        bands_qp.append(temp[:, 3])
        
    bands_dft = np.array(bands_dft) * eV2ry  # shape (Nbnds, Nk)
    bands_qp = np.array(bands_qp) * eV2ry  # shape (Nbnds, Nk)
    return bands_dft, bands_qp, Kpoints, Nk, band_indexes


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import sys
    import os
    import h5py
    import numpy as np

    parser = argparse.ArgumentParser(description='Compute second-order e-ph coefficients.')
    parser.add_argument('--elph', default='elph_coeffs.h5', dest='elph_coeffs_file_to_be_loaded',
                        help='Path to the elph HDF5 file (default: elph_coeffs.h5)')
    parser.add_argument('--eqp', default='eqp.dat', dest='eqp_file',
                        help='Path to the eqp.dat file (default: eqp.dat)')
    parser.add_argument('--nval', default=1, type=int, dest='Nval',
                        help='Band index of valence band maximum, 1-based (default: 1)')
    args = parser.parse_args()

    elph_coeffs_file_to_be_loaded = args.elph_coeffs_file_to_be_loaded
    eqp_file = args.eqp_file
    Nval = args.Nval

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../main')))
    from excited_forces_m import load_elph_coeffs_hdf5
    from excited_forces_m import save_elph_coeffs_hdf5

    bands_dft, bands_qp, Kpoints, Nk_in_eqp_file, band_indexes = read_eqp_dat_file(eqp_file)
    ival = np.where(band_indexes == Nval)[0][0] + 1
    E_val_dft  = bands_dft[:ival]   # shape (Nv, Nk)
    E_cond_dft = bands_dft[ival:]   # shape (Nc, Nk)
    E_val_qp   = bands_qp[:ival]
    E_cond_qp  = bands_qp[ival:]
    
    print('E_cond_qp.shape:', E_cond_qp.shape)
    print('E_val_qp.shape:', E_val_qp.shape)

    (elph_cond, elph_val,
     elph_cond_not_renorm, elph_val_not_renorm,
     Displacements, elph_fine_a_la_bgw,
     no_renorm_elph, Kpoints_in_elph_file_frac) = load_elph_coeffs_hdf5(elph_coeffs_file_to_be_loaded)
    
    print('Loaded elph coefficients from file:', elph_coeffs_file_to_be_loaded)
    print('elph_cond shape:', elph_cond.shape) # shape (nmodes, Nk, Nc, Nc)
    print('elph_val shape:', elph_val.shape)   # shape (nmodes, Nk, Nv, Nv)
    Nk_in_elph_file = elph_cond.shape[1]
    
    if Nk_in_elph_file != Nk_in_eqp_file:
        print(f'Error: Number of k-points in elph file ({Nk_in_elph_file}) does not match number of k-points in eqp file ({Nk_in_eqp_file}).')
        sys.exit(1)
        
    if elph_cond.shape[2] > E_cond_qp.shape[0]:
        print(f'Error: Number of conduction bands in elph file ({elph_cond.shape[2]}) is greater than the number of conduction bands in eqp file ({E_cond_qp.shape[0]}).')
        sys.exit(1)
    elif elph_cond.shape[2] < E_cond_qp.shape[0]:
        print(f'Number of conduction bands in elph file ({elph_cond.shape[2]}) is less than the number of conduction bands in eqp file ({E_cond_qp.shape[0]}). Only the first {elph_cond.shape[2]} conduction bands will be used for the calculation.')

    if elph_val.shape[2] > E_val_qp.shape[0]:
        print(f'Error: Number of valence bands in elph file ({elph_val.shape[2]}) is greater than the number of valence bands in eqp file ({E_val_qp.shape[0]}).')
        sys.exit(1)
    elif elph_val.shape[2] < E_val_qp.shape[0]:
        print(f'Number of valence bands in elph file ({elph_val.shape[2]}) is less than the number of valence bands in eqp file ({E_val_qp.shape[0]}). Only the first {elph_val.shape[2]} valence bands will be used for the calculation.')

    # shape of elph_* arrays: (nmodes, nk, nbnds, nbnds)

    g2_cond = compute_second_order_elph(elph_cond, E_cond_qp, Displacements)
    g2_val  = compute_second_order_elph(elph_val,  E_val_qp,  Displacements)
    
    g2_cond_not_renorm = compute_second_order_elph(elph_cond_not_renorm, E_cond_dft, Displacements)
    g2_val_not_renorm  = compute_second_order_elph(elph_val_not_renorm,  E_val_dft,  Displacements)
    print('g2_cond shape:', g2_cond.shape)
    print('g2_val  shape:', g2_val.shape)
    
    # save the results in an hdf5 file
    # save in the same format as the original elph coefficients, i.e. with the same dimensions and indexing
    # so it can be read by excited_state.py to compute second derivatives of exciton-phonon matrix elements 
    
    save_elph_coeffs_hdf5(g2_cond, g2_val, g2_cond_not_renorm, g2_val_not_renorm, 
                          Displacements, elph_fine_a_la_bgw, no_renorm_elph, 
                          Kpoints_in_elph_file_frac, '2nd_derivative_elph_coeffs.h5')

    print('Second-order e-ph coefficients computed and saved to second_derivative_elph_coeffs.h5')
