

import numpy as np

TOL_ZERO = 1e-6  # threshold below which states are considered degenerate

def compute_second_order_elph(elph, E, Displacements):
    """
    Compute second-order electron-phonon coupling coefficients via second-order
    perturbation theory:

        g_beta_knm^(2) = - sum_{alpha,l} g_alpha_knl * g_beta_klm / (E_kn - E_kl)
                         - sum_{alpha,l} g_beta_knl * g_alpha_klm / (E_km - E_kl)

    The sum over alpha runs over all cartesian (atom-direction) degrees of freedom.
    If `elph` is in the normal-mode basis it is rotated to cartesian first via
    `Displacements`.  The returned array is in the same cartesian basis (first
    index = displacement-pattern / atom-direction index).

    Parameters
    ----------
    elph : np.ndarray, shape (nmodes, nkpoints, nbands, nbands)
        Electron-phonon coupling coefficients, possibly in normal-mode basis.
        elph[mu, k, n, m] = g_{mu, k, n->m}
    E : np.ndarray, shape (nbands, nkpoints)
        Single-particle energies.  E[n, k] = energy of band n at k-point k.
    Displacements : np.ndarray, shape (nmodes, nmodes)
        Displacement patterns.  Displacements[mu, :] = [x1,y1,z1, x2,y2,z2, ...]
        Row mu gives the cartesian displacements of all atoms for normal mode mu.
        Used to rotate elph from normal-mode to cartesian basis.

    Returns
    -------
    second_der_elph : np.ndarray, shape (nmodes, nkpoints, nbands, nbands)
        Second-order e-ph coefficients indexed by cartesian displacement (alpha).
    """
    
    # --- rotate elph to cartesian (atom-direction) basis -------------------
    # g_cart[beta, k, n, l] = sum_mu  D[mu, beta] * elph[mu, k, n, l]
    g_cart = np.einsum('mb,mknl->bknl', Displacements, elph)

    # Nmodes, Nk, nbands, _ = elph.shape
    # term1 = np.zeros_like(g_cart)
    # term2 = np.zeros_like(g_cart)
    
    # for beta in range(Nmodes):
    #     for ik in range(Nk):
    #         for n in range(nbands):
    #             for m in range(nbands):
    #                 sum1 = 0.0
    #                 sum2 = 0.0
    #                 for l in range(nbands):
    #                     dE_nl = E[n, ik] - E[l, ik]
    #                     dE_ml = E[m, ik] - E[l, ik]
    #                     if np.abs(dE_nl) > TOL_ZERO:
    #                         sum1 += g_cart[beta, ik, n, l] * g_cart[beta, ik, l, m] / dE_nl
    #                     if np.abs(dE_ml) > TOL_ZERO:
    #                         sum2 += g_cart[beta, ik, n, l] * g_cart[beta, ik, l, m] / dE_ml
    #                 term1[beta, ik, n, m] = -sum1
    #                 term2[beta, ik, n, m] = -sum2

    # dE[k, i, j] = E[i, k] - E[j, k]   (E has shape (nbands, nk))
    dE = E.T[:, :, None] - E.T[:, None, :]  # (nk, nbands, nbands)
    mask = np.abs(dE) < TOL_ZERO
    dE_safe = np.where(mask, 1.0, dE)

    # term1[b, k, n, m] = -sum_l  g_cart[b,k,n,l] / dE[k,n,l]  * g_cart[b,k,l,m]
    weighted1 = np.where(mask, 0.0, g_cart / dE_safe[np.newaxis])  # (b, nk, nbands, nbands)
    term1 = -np.einsum('bknl,bklm->bknm', weighted1, g_cart)

    # term2[b, k, n, m] = -sum_l  g_cart[b,k,n,l] * g_cart[b,k,l,m] / dE[k,m,l]
    # dE[k,m,l] == dE.T[k,l,m]
    dE_T = dE.transpose(0, 2, 1)                                    # (nk, nbands, nbands)
    mask_T = np.abs(dE_T) < TOL_ZERO
    dE_T_safe = np.where(mask_T, 1.0, dE_T)
    weighted2 = np.where(mask_T, 0.0, g_cart / dE_T_safe[np.newaxis])  # (b, nk, l, m)
    term2 = -np.einsum('bknl,bklm->bknm', g_cart, weighted2)

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
    return np.array(bands_dft), np.array(bands_qp), Kpoints, Nk, band_indexes


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

    bands_dft, bands_qp, Kpoints, Nk, band_indexes = read_eqp_dat_file(eqp_file)
    ival = np.where(band_indexes == Nval)[0][0] + 1
    E_val_dft  = bands_dft[:ival]   # shape (Nv, Nk)
    E_cond_dft = bands_dft[ival:]   # shape (Nc, Nk)
    E_val_qp   = bands_qp[:ival]
    E_cond_qp  = bands_qp[ival:]

    (elph_cond, elph_val,
     elph_cond_not_renorm, elph_val_not_renorm,
     Displacements, elph_fine_a_la_bgw,
     no_renorm_elph, Kpoints_in_elph_file_frac) = load_elph_coeffs_hdf5(elph_coeffs_file_to_be_loaded)

    # shape of elph_* arrays: (nmodes, nk, nbnds, nbnds)

    g2_cond = compute_second_order_elph(elph_cond, E_cond_qp, Displacements)
    g2_val  = compute_second_order_elph(elph_val,  E_val_qp,  Displacements)
    
    g2_cond_not_renorm = compute_second_order_elph(elph_cond_not_renorm, E_cond_dft, Displacements)
    g2_val_not_renorm  = compute_second_order_elph(elph_val_not_renorm,  E_val_dft,  Displacements)
    print('g2_cond shape:', g2_cond.shape)
    print('g2_val  shape:', g2_val.shape)
    
    # save the results in an hdf5 file
    with h5py.File('second_derivative_elph_coeffs.h5', 'w') as hf:
        hf.create_dataset('g2_cond', data=g2_cond)
        hf.create_dataset('g2_val', data=g2_val)
        hf.create_dataset('g2_cond_not_renorm', data=g2_cond_not_renorm)
        hf.create_dataset('g2_val_not_renorm', data=g2_val_not_renorm)
        hf.create_dataset('Displacements', data=Displacements)
        hf.create_dataset('Kpoints_in_elph_file_frac', data=Kpoints_in_elph_file_frac)
        
    print('Second-order e-ph coefficients computed and saved to second_derivative_elph_coeffs.h5')
