import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cross_decomposition import PLSRegression
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10

# ---------- Helper Functions ----------

def parse_spectrum(file):
    content = file.getvalue().decode('utf-8', errors='ignore').splitlines()
    data = []
    for line in content:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'[\s,;\t]+', line)
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                data.append((x, y))
            except ValueError:
                continue
    if data:
        df = pd.DataFrame(data, columns=["Wavelength", "Absorbance"])
        df.sort_values(by="Wavelength", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    else:
        return None

def plot_spectra(spectra, title):
    p = figure(title=title, width=700, height=400, x_axis_label='Wavelength (nm)', y_axis_label='Absorbance')
    colors = Category10[10]
    for idx, (name, df) in enumerate(spectra.items()):
        source = ColumnDataSource(df)
        p.line('Wavelength', 'Absorbance', source=source, legend_label=name, line_width=2, color=colors[idx % len(colors)])
    p.legend.click_policy = "hide"
    return p

def calculate_sample_concentrations(stock_concentrations, volumes, total_volume):
    concentrations = (stock_concentrations.values * volumes.values) / total_volume
    return pd.DataFrame(concentrations, columns=stock_concentrations.index)

# ---------- Streamlit App ----------

st.set_page_config(page_title="UV-Vis Chemometrics App", layout="wide")

st.title("UV-Vis Spectra Chemometric Analysis")
st.info("Upload training spectra for multivariate model calibration. Define stock solution concentrations (mg/mL) and mixture volumes (mL).")

# Sidebar Inputs
st.sidebar.header("Inputs")
uploaded_train_files = st.sidebar.file_uploader("Upload Training Spectra", accept_multiple_files=True, type=['csv', 'txt'])
uploaded_measurement_files = st.sidebar.file_uploader("Upload Measurement Spectra (Triplicates)", accept_multiple_files=True, type=['csv', 'txt'])

num_components = st.sidebar.number_input("Number of Components", min_value=1, max_value=4, value=2)
train_button = st.sidebar.button("Train Model")
analyze_button = st.sidebar.button("Analyze Measurements")

# Data Handling
train_spectra = {}
measurement_spectra = {}

if uploaded_train_files:
    for file in uploaded_train_files:
        df = parse_spectrum(file)
        if df is not None:
            train_spectra[file.name] = df
        else:
            st.warning(f"File {file.name} could not be parsed and was skipped.")

if uploaded_measurement_files:
    for file in uploaded_measurement_files:
        df = parse_spectrum(file)
        if df is not None:
            measurement_spectra[file.name] = df
        else:
            st.warning(f"File {file.name} could not be parsed and was skipped.")

if train_spectra:
    st.subheader("Training Spectra")
    st.bokeh_chart(plot_spectra(train_spectra, "Training Spectra"), use_container_width=True)

    # Create Component Table
    st.subheader("Stock Solution Concentrations (mg/mL)")
    components = [f"Component {i+1}" for i in range(num_components)]
    stock_concentrations_df = pd.DataFrame({
        "Component": components,
        "Stock Concentration (mg/mL)": [1.0 for _ in range(num_components)]
    })
    stock_concentrations_df = st.data_editor(stock_concentrations_df, num_rows="fixed", key="stock_conc")

    # Create Mixing Table with real sample names
    st.subheader("Mixture Volumes (mL)")
    sample_names = list(train_spectra.keys())
    columns = components + ["Water"]
    mixing_df = pd.DataFrame(0.0, index=sample_names, columns=columns)
    mixing_df = st.data_editor(mixing_df, key="mixing_volumes")

# --------------- Run Analysis ------------------

if train_button:
    if not train_spectra:
        st.error("Please upload training spectra!")
    else:
        st.subheader("Training Model...")
        try:
            wavelengths = np.linspace(800, 200, num=601)  # Uniform grid

            # Interpolate training spectra
            X_train = np.array([
                np.interp(wavelengths, df['Wavelength'], df['Absorbance']) for df in train_spectra.values()
            ])

            # Calculate concentrations
            stock_conc_values = pd.Series(stock_concentrations_df["Stock Concentration (mg/mL)"].values, index=components)
            volumes_only = mixing_df[components]
            total_volumes = volumes_only.sum(axis=1) + mixing_df["Water"]
            y_train = calculate_sample_concentrations(stock_conc_values, volumes_only, total_volumes.values[:, np.newaxis])

            # PLS model
            pls = PLSRegression(n_components=min(num_components, X_train.shape[0]-1, X_train.shape[1]))
            pls.fit(X_train, y_train)

            st.success("Model trained successfully!")

        except Exception as e:
            st.error(f"Error during training: {e}")

if analyze_button:
    if not (train_spectra and measurement_spectra):
        st.error("Please upload both training and measurement spectra!")
    else:
        try:
            st.subheader("Running Chemometric Analysis...")

            wavelengths = np.linspace(800, 200, num=601)

            X_train = np.array([
                np.interp(wavelengths, df['Wavelength'], df['Absorbance']) for df in train_spectra.values()
            ])

            stock_conc_values = pd.Series(stock_concentrations_df["Stock Concentration (mg/mL)"].values, index=components)
            volumes_only = mixing_df[components]
            total_volumes = volumes_only.sum(axis=1) + mixing_df["Water"]
            y_train = calculate_sample_concentrations(stock_conc_values, volumes_only, total_volumes.values[:, np.newaxis])

            pls = PLSRegression(n_components=min(num_components, X_train.shape[0]-1, X_train.shape[1]))
            pls.fit(X_train, y_train)

            X_measure = np.array([
                np.interp(wavelengths, df['Wavelength'], df['Absorbance']) for df in measurement_spectra.values()
            ])
            y_pred = pls.predict(X_measure)

            # Display spectra
            st.subheader("Measurement Spectra")
            st.bokeh_chart(plot_spectra(measurement_spectra, "Measurement Spectra"), use_container_width=True)

            # Results
            pred_df = pd.DataFrame(y_pred, columns=components, index=list(measurement_spectra.keys()))
            pred_mean = pred_df.mean(axis=0)
            pred_std = pred_df.std(axis=0)

            st.subheader("Predicted Concentrations (mg/mL)")
            st.write("Mean Concentration for each component (from triplicates):")
            st.dataframe(pred_mean)
            st.write("Standard Deviation of Concentrations:")
            st.dataframe(pred_std)

        except Exception as e:
            st.error(f"Error during analysis: {e}")
