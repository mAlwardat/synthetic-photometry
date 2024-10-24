import pysynphot as S
import numpy as np
import astropy.units as u
import pandas as pd

filters = {
    'johnson.u': {'wavelength': np.array([3000, 3500, 4000]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 1500 * u.Jy},
    'johnson.b': {'wavelength': np.array([3500, 4500, 5500]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 4260 * u.Jy},
    'johnson.v': {'wavelength': np.array([4500, 5500, 6500]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 3640 * u.Jy},
    'cousins.r': {'wavelength': np.array([5500, 6500, 7500]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 3080 * u.Jy},
    'cousins.i': {'wavelength': np.array([6500, 7500, 8500]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 2550 * u.Jy},
    'stromgren.u': {'wavelength': np.array([3000, 3500, 4000]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 1810 * u.Jy},
    'stromgren.v': {'wavelength': np.array([3500, 4500, 5500]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 4260 * u.Jy},
    'stromgren.b': {'wavelength': np.array([4500, 5500, 6500]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 3640 * u.Jy},
    '2MASS.J': {'wavelength': np.array([11000, 12500, 14000]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 1594 * u.Jy},
    '2MASS.H': {'wavelength': np.array([14000, 16500, 19000]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 1024 * u.Jy},
    'gaia.G': {'wavelength': np.array([3300, 6400, 10500]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 3631 * u.Jy},
    'gaia.BP': {'wavelength': np.array([3300, 5100, 6800]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 6400 * u.Jy},
    'gaia.RP': {'wavelength': np.array([6300, 7700, 10500]) * u.AA, 'transmission': np.array([0.0, 1.0, 0.0]), 'zero_point': 2550 * u.Jy},
}

# Function to read spectrum data
def read_spectrum(uploaded_file):
    if uploaded_file.type == 'text/csv':
        data = pd.read_csv(uploaded_file, header=0, dtype='float64')
    else:
        data = pd.read_csv(uploaded_file, delimiter="\\t", header=None)
    if data.shape[1] > 2 and data.shape[1] % 2 == 0:
        data.drop(data.columns[0], axis=1, inplace=True)
        data.drop(data.columns[[i for i in range(
            2, data.shape[1], 2)]], axis=1, inplace=True)
        data.columns = data.iloc[0]
        data.drop(data.index[0], inplace=True)
        data.reset_index(drop=True, inplace=True)
    elif data.shape[1] == 2:
        data.columns = ['wavelength', 'flux']

    wavelength = data.iloc[:, 0]
    flux = data.iloc[:, 1:]
    wavelength = wavelength.values * u.AA
    return wavelength, flux

# Step 3: Define filters and zero points


def calculate_magnitudes(wavelength, flux):
    flux *= (u.erg / (u.cm**2 * u.s * u.AA))
    spectrum = S.ArraySpectrum(
        wave=wavelength.value, flux=flux.value, waveunits='angstrom', fluxunits='flam')
    magnitudes = {}
    for filter_id, filter_data in filters.items():
        bandpass = S.ArrayBandpass(
            filter_data['wavelength'].value, filter_data['transmission'], waveunits='angstrom')
        # Define binset for the bandpass
        bandpass.binset = spectrum.wave
        observation = S.Observation(spectrum, bandpass)
        flux_in_band = observation.effstim(
            'flam') * (u.erg / (u.cm**2 * u.s * u.AA))
        zero_point_flux = filter_data['zero_point'].to(
            u.erg / (u.cm**2 * u.s * u.AA), equivalencies=u.spectral_density(filter_data['wavelength'].mean()))
        mag = -2.5 * np.log10((flux_in_band / zero_point_flux).value)
        magnitudes[filter_id] = mag
    return magnitudes
