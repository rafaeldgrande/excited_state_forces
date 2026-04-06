
import numpy as np
import h5py
import matplotlib.pyplot as plt

def bose_einstein_distribution(energy, temperature):
    """Calculate the Bose-Einstein distribution for a given energy and temperature."""
    return 1 / (np.exp(energy / (k_B * temperature)) - 1)

# constants
k_B = 8.617333262145e-5  # Boltzmann constant in eV/K
c = 2.99792458e-4  # Speed of light in eV*cm
h = 4.135667696e-15  # Planck's constant in eV*s
rec_cm_to_eV = 1.239841984e-4  # Conversion factor from cm^-1 to eV


log_scale = False  # Set to True to plot Raman intensity on a logarithmic scale, False for linear scale
T = 300 # Temperature in Kelvin
susceptibility_tensors_file = 'susceptibility_tensors.h5'
freqs_file = 'freqs.dat'  # File containing phonon frequencies in cm^-1
cart_dir = ['x', 'y', 'z']  # Cartesian directions for the susceptibility tensor components

freqs_rec_cm = np.loadtxt(freqs_file)  # Load phonon frequencies in cm^-1
freqs_eV = freqs_rec_cm * rec_cm_to_eV  # Convert frequencies

with h5py.File(susceptibility_tensors_file, 'r') as f:
    excitaion_energies = f['excitation_energies'][:]
    alpha_tensor_d2 = f['alpha_tensor_d2'][:] # shape (3, 3, Nmodes, Ndata)
    alpha_tensor_d3 = f['alpha_tensor_d3'][:] # shape (3, 3, Nmodes, Ndata)
    
Nmodes = alpha_tensor_d2.shape[2]

# unpolarized tensors
alpha_d2 = (alpha_tensor_d2[0, 0] + alpha_tensor_d2[1, 1] + alpha_tensor_d2[2, 2]) / 3.0

gamma2_d2 = ( 0.5 * (np.abs(alpha_tensor_d2[0, 0] - alpha_tensor_d2[1,1])**2 +
                     np.abs(alpha_tensor_d2[1, 1] - alpha_tensor_d2[2,2])**2 +
                     np.abs(alpha_tensor_d2[2, 2] - alpha_tensor_d2[0,0])**2  ) +
             3/4 *  (np.abs(alpha_tensor_d2[0, 1] + alpha_tensor_d2[1,0])**2 +
                     np.abs(alpha_tensor_d2[0, 2] + alpha_tensor_d2[2,0])**2 +
                     np.abs(alpha_tensor_d2[1, 2] + alpha_tensor_d2[2,1])**2 ) )

delta2_d2 = 3/4 * ((np.abs(alpha_tensor_d2[0, 1] - alpha_tensor_d2[1,0])**2 +
                     np.abs(alpha_tensor_d2[0, 2] - alpha_tensor_d2[2,0])**2 +
                     np.abs(alpha_tensor_d2[1, 2] - alpha_tensor_d2[2,1])**2 ))

alpha_d3 = (alpha_tensor_d3[0, 0] + alpha_tensor_d3[1, 1] + alpha_tensor_d3[2, 2]) / 3.0

gamma2_d3 = ( 0.5 * (np.abs(alpha_tensor_d3[0, 0] - alpha_tensor_d3[1,1])**2 +
                     np.abs(alpha_tensor_d3[1, 1] - alpha_tensor_d3[2,2])**2 +
                     np.abs(alpha_tensor_d3[2, 2] - alpha_tensor_d3[0,0])**2  ) +
             3/4 *  (np.abs(alpha_tensor_d3[0, 1] + alpha_tensor_d3[1,0])**2 +
                     np.abs(alpha_tensor_d3[0, 2] + alpha_tensor_d3[2,0])**2 +
                     np.abs(alpha_tensor_d3[1, 2] + alpha_tensor_d3[2,1])**2 ) )

delta2_d3 = 3/4 * ((np.abs(alpha_tensor_d3[0, 1] - alpha_tensor_d3[1,0])**2 +
                     np.abs(alpha_tensor_d3[0, 2] - alpha_tensor_d3[2,0])**2 +
                     np.abs(alpha_tensor_d3[1, 2] - alpha_tensor_d3[2,1])**2 ))

tensor_d2_unpolarized = 45 * alpha_d2**2 + 7 * gamma2_d2 + 5 * delta2_d2
tensor_d3_unpolarized = 45 * alpha_d3**2 + 7 * gamma2_d3 + 5 * delta2_d3

def raman_cross_section(excitation, intensity, freqs_eV, T, log_scale=False):
    cross_section = excitation**3 * np.abs(intensity)**2 * (1 + bose_einstein_distribution(freqs_eV, T)) * freqs_eV
    if log_scale == False:
        return cross_section
    else:
        return np.log10(cross_section + 1e-2)  # add small value to avoid log(0)
    
for imode in range(Nmodes):
    n_phonon = bose_einstein_distribution(freqs_eV[imode], T)
    print(f'Phonon mode {imode+1} / {Nmodes}: frequency = {freqs_rec_cm[imode]:.4f} cm-1, occupation number = {n_phonon:.4f}')
    f, axs = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
    f.suptitle(f'Phonon mode {imode+1} at T={T} K and frequency {freqs_rec_cm[imode]:.2f} cm-1', fontsize=12)
    
    for ialpha in range(3):
        for ibeta in range(3):
            d2 = alpha_tensor_d2[ialpha, ibeta, imode, :]
            d3 = alpha_tensor_d3[ialpha, ibeta, imode, :]
            
            raman_d2 = raman_cross_section(excitaion_energies, d2, freqs_eV[imode], T, log_scale=log_scale)
            raman_d3 = raman_cross_section(excitaion_energies, d3, freqs_eV[imode], T, log_scale=log_scale)
            
            plt.sca(axs[ialpha, ibeta])
            plt.title(f'{cart_dir[ialpha]}{cart_dir[ibeta]} component')
            plt.plot(excitaion_energies, raman_d2, label=f'd2:')
            plt.plot(excitaion_energies, raman_d3, label=f'd3+d2', linestyle='dashed')
    
    plt.xlabel('Excitation Energy (eV)')
    plt.ylabel('Raman Intensity (a.u.)')
    # plt.legend()
    plt.savefig(f'raman_spectrum_mode_{imode+1}.png')
    plt.close()
    
min_vib_freq = np.min(freqs_rec_cm)
max_vib_freq = np.max(freqs_rec_cm)

# 2d map of Raman intensity as a function of excitation energy and phonon frequency

# phonon_broadening = 10.0  # Lorentzian FWHM in cm^-1
phonon_broadening = (max_vib_freq - min_vib_freq) / 50.0  # Set broadening to 1/50 of the total phonon frequency range
freq_axis = np.linspace(min_vib_freq - 5*phonon_broadening, max_vib_freq + 5*phonon_broadening, 500)
excitation_grid, freq_grid = np.meshgrid(excitaion_energies, freq_axis)

for ialpha in range(3):
    for ibeta in range(3):

        raman_intensity_map_d2 = np.zeros_like(excitation_grid)
        raman_intensity_map_d3 = np.zeros_like(excitation_grid)
        for imode in range(Nmodes):
            d2 = alpha_tensor_d2[ialpha, ibeta, imode, :]
            d3 = alpha_tensor_d3[ialpha, ibeta, imode, :]

            raman_d2 = raman_cross_section(excitaion_energies, d2, freqs_eV[imode], T, log_scale=log_scale)
            raman_d3 = raman_cross_section(excitaion_energies, d3, freqs_eV[imode], T, log_scale=log_scale)

            # Lorentzian weight along phonon frequency axis, centered at mode frequency
            gamma = phonon_broadening / 2
            lorentzian = gamma**2 / ((freq_axis - freqs_rec_cm[imode])**2 + gamma**2)

            raman_intensity_map_d2 += lorentzian[:, np.newaxis] * raman_d2[np.newaxis, :]
            raman_intensity_map_d3 += lorentzian[:, np.newaxis] * raman_d3[np.newaxis, :]

        f, axs = plt.subplots(ncols=2, figsize=(15, 8), sharex=True, sharey=True)
        
        plt.sca(axs[0])
        plt.title(f'{cart_dir[ialpha]}{cart_dir[ibeta]} component - d2', fontsize=18)
        plt.pcolormesh(excitation_grid, freq_grid, raman_intensity_map_d2, shading='auto')
        plt.colorbar(label='Raman Intensity (a.u.)')
        plt.xlabel('Excitation Energy (eV)')
        plt.ylabel('Phonon Frequency (cm$^{-1}$)')
        
        plt.sca(axs[1])
        plt.title(f'{cart_dir[ialpha]}{cart_dir[ibeta]} component - d3+d2', fontsize=18)
        plt.pcolormesh(excitation_grid, freq_grid, raman_intensity_map_d3, shading='auto')
        plt.colorbar(label='Raman Intensity (a.u.)')
        plt.xlabel('Excitation Energy (eV)')
        plt.ylabel('Phonon Frequency (cm$^{-1}$)')
        
        plt.savefig(f'raman_intensity_map_{cart_dir[ialpha]}{cart_dir[ibeta]}.png', dpi=300)
        plt.close()
        

raman_intensity_map_d2 = np.zeros_like(excitation_grid)
raman_intensity_map_d3 = np.zeros_like(excitation_grid)
for imode in range(Nmodes):
    d2 = tensor_d2_unpolarized[imode, :]
    d3 = tensor_d3_unpolarized[imode, :]

    raman_d2 = raman_cross_section(excitaion_energies, d2, freqs_eV[imode], T, log_scale=log_scale)
    raman_d3 = raman_cross_section(excitaion_energies, d3, freqs_eV[imode], T, log_scale=log_scale)

    # Lorentzian weight along phonon frequency axis, centered at mode frequency
    gamma = phonon_broadening / 2
    lorentzian = gamma**2 / ((freq_axis - freqs_rec_cm[imode])**2 + gamma**2)

    raman_intensity_map_d2 += lorentzian[:, np.newaxis] * raman_d2[np.newaxis, :]
    raman_intensity_map_d3 += lorentzian[:, np.newaxis] * raman_d3[np.newaxis, :]        

f, axs = plt.subplots(ncols=2, figsize=(15, 8), sharex=True, sharey=True)

plt.sca(axs[0])
plt.title(f'Unpolarized - d2', fontsize=18)
plt.pcolormesh(excitation_grid, freq_grid, raman_intensity_map_d2, shading='auto')
plt.colorbar(label='Raman Intensity (a.u.)')
plt.xlabel('Excitation Energy (eV)')
plt.ylabel('Phonon Frequency (cm$^{-1}$)')

plt.sca(axs[1])
plt.title(f'Unpolarized - d3+d2', fontsize=18)
plt.pcolormesh(excitation_grid, freq_grid, raman_intensity_map_d3, shading='auto')
plt.colorbar(label='Raman Intensity (a.u.)')
plt.xlabel('Excitation Energy (eV)')
plt.ylabel('Phonon Frequency (cm$^{-1}$)')

plt.savefig(f'raman_intensity_map_unpolarized.png', dpi=300)
plt.close()


    
print('Raman spectra calculated and saved as PNG files.')

