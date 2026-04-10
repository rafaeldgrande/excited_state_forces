
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import h5py
from pathlib import Path

config_dir = Path(__file__).parent.parent / 'presentation.mplstyle'
plt.style.use(config_dir)

h5_file = 'exciton_phonon_couplings.h5'
with h5py.File(h5_file, 'r') as hf:
    ibl_exc_ph = hf['rpa_diag'][:]  # shape: (Nmodes, Nexciton, Nexciton)
    dgs_exc_ph = hf['rpa_offdiag'][:]  # shape: (Nmodes, Nexciton, Nexciton)


Nmodes = ibl_exc_ph.shape[0]
Nexc_plot = ibl_exc_ph.shape[1]

ibl_exc_ph_diag = ibl_exc_ph[:, np.arange(Nexc_plot), np.arange(Nexc_plot)]  # shape: (Nmodes, Nexciton)
dgs_exc_ph_diag = dgs_exc_ph[:, np.arange(Nexc_plot), np.arange(Nexc_plot)]  # shape: (Nmodes, Nexciton)

eigvals_file = 'eigenvalues_b1.dat'
exc_energies = np.loadtxt(eigvals_file)[:Nexc_plot, 0]  # shape: (Nexciton,)

exc_energies_diff = np.zeros((Nexc_plot, Nexc_plot))
for i in range(Nexc_plot):
    for j in range(Nexc_plot):
        exc_energies_diff[i, j] = exc_energies[i] - exc_energies[j]

# with PdfPages('report_exc_ph.pdf') as pdf:
for imode in range(Nmodes):
    
    print(f'Plotting data for mode {imode+1} of {Nmodes}')
    
    f, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10), sharex=False, sharey=False)
    f.suptitle(f'Phonon mode {imode+1}', fontsize=12)

    im00 = axs[0,0].imshow(np.abs(ibl_exc_ph[imode]), origin='lower', aspect='auto')
    f.colorbar(im00, ax=axs[0,0])
    axs[0,0].set_title('IBL')

    im01 = axs[0,1].imshow(np.abs(dgs_exc_ph[imode]), origin='lower', aspect='auto')
    f.colorbar(im01, ax=axs[0,1])
    axs[0,1].set_title('DGS')

    axs[1,0].scatter(np.abs(exc_energies_diff.flatten()), np.abs(ibl_exc_ph[imode].flatten()), s=10, alpha=0.5)
    axs[1,0].set_xlabel(r'$\Delta \Omega$ (eV)')
    axs[1,0].set_ylabel(r'IBL (eV/$\rm{\AA}$)')

    axs[1,1].scatter(np.abs(exc_energies_diff.flatten()), np.abs(dgs_exc_ph[imode].flatten()), s=10, alpha=0.5)
    axs[1,1].set_xlabel(r'$\Delta \Omega$ (eV)')
    axs[1,1].set_ylabel(r'DGS (eV/$\rm{\AA}$)')

    # f.tight_layout()
    # pdf.savefig(f)
    # plt.close(f)
    
    plt.savefig(f'report_exc_ph_mode_{imode+1}.png', dpi=300)
    plt.close()
    

