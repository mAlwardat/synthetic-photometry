import streamlit as st
import utils
import pandas as pd

st.title("Spectral Photometry")

uploaded_file = st.file_uploader(
    label='Upload File Here', type=['dat', 'csv'])

output = {"Filters": ['johnson.u', 'johnson.b', 'johnson.v', 'cousins.r', 'cousins.i',
                      'stromgren.u', 'stromgren.v', 'stromgren.b', '2MASS.J', '2MASS.H', 'gaia.G', 'gaia.BP', 'gaia.RP']}

if uploaded_file is not None:
    wavelength, flux = utils.read_spectrum(uploaded_file)
    output['Total Flux'] = list(utils.calculate_magnitudes(
        wavelength, flux.iloc[:, 0].values).values())

    if flux.shape[1] > 1:
        for i in range(1, flux.shape[1]):
            output[f'Star {i} Flux'] = list(utils.calculate_magnitudes(
                wavelength, flux.iloc[:, i].values).values())
            
    output_df = pd.DataFrame(output).T
    output_df.columns = output_df.iloc[0]
    output_df.drop(output_df.index[0], inplace=True)
    st.write(output_df)