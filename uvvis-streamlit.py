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

st.set_page_config(page_title="UV-Vis PLS Analyse", layout="wide")

st.title("UV-Vis Spektralanalyse mit PLS")
st.info("Trainiere ein Modell zur Vorhersage von Konzentrationen basierend auf UV-Vis Spektren.")

# Sidebar Inputs
st.sidebar.header("Einstellungen")

num_components = st.sidebar.number_input("Anzahl chemischer Komponenten", min_value=1, max_value=10, value=3)
pls_components = st.sidebar.number_input("Anzahl PLS-Komponenten", min_value=1, max_value=num_components, value=num_components - 1)

uploaded_train_files = st.sidebar.file_uploader("Trainingsspektren hochladen", accept_multiple_files=True, type=['csv', 'txt'])
uploaded_measurement_files = st.sidebar.file_uploader("Messspektren (Triplikate) hochladen", accept_multiple_files=True, type=['csv', 'txt'])

train_button = st.sidebar.button("Modell trainieren")
analyze_button = st.sidebar.button("Messung analysieren")

# Daten einlesen
train_spectra = {}
measurement_spectra = {}

if uploaded_train_files:
    for file in uploaded_train_files:
        df = parse_spectrum(file)
        if df is not None:
            train_spectra[file.name] = df
        else:
            st.warning(f"{file.name} konnte nicht gelesen werden.")

if uploaded_measurement_files:
    for file in uploaded_measurement_files:
        df = parse_spectrum(file)
        if df is not None:
            measurement_spectra[file.name] = df
        else:
            st.warning(f"{file.name} konnte nicht gelesen werden.")

if train_spectra:
    st.subheader("Trainingsspektren")
    st.bokeh_chart(plot_spectra(train_spectra, "Trainingsspektren"), use_container_width=True)

    components = [f"Komponente {i+1}" for i in range(num_components)]

    st.subheader("Stammlösungs-Konzentrationen (mg/mL)")
    stock_conc_df = pd.DataFrame({
        "Komponente": components,
        "Konzentration (mg/mL)": [1.0 for _ in range(num_components)]
    })
    stock_conc_df = st.data_editor(stock_conc_df, num_rows="fixed", key="stock_conc")

    st.subheader("Mischverhältnisse (Volumen in mL)")
    sample_names = list(train_spectra.keys())
    mixing_columns = components + ["Wasser"]
    mixing_df = pd.DataFrame(0.0, index=sample_names, columns=mixing_columns)
    mixing_df = st.data_editor(mixing_df, key="mixing_volumes")

# --------- Modell Training -----------

if train_button:
    if not train_spectra:
        st.error("Bitte Trainingsspektren hochladen!")
    else:
        try:
            st.subheader("Trainiere Modell...")

            wavelengths = np.linspace(800, 200, num=601)
            X_train = np.array([
                np.interp(wavelengths, df['Wavelength'], df['Absorbance']) for df in train_spectra.values()
            ])

            # Konzentrationen berechnen
            stock_conc_series = pd.Series(stock_conc_df["Konzentration (mg/mL)"].values, index=components)
            volumes_only = mixing_df[components]
            total_volumes = volumes_only.sum(axis=1) + mixing_df["Wasser"]

            if not np.allclose(total_volumes.values, total_volumes.iloc[0], atol=0.01):
                st.error("Nicht alle Mischungen ergeben dasselbe Gesamtvolumen!")
            else:
                y_train = calculate_sample_concentrations(stock_conc_series, volumes_only, total_volumes.values[:, np.newaxis])
                pls = PLSRegression(n_components=pls_components)
                pls.fit(X_train, y_train)
                st.session_state["pls_model"] = pls

                st.success("Modell erfolgreich trainiert!")

        except Exception as e:
            st.error(f"Fehler beim Training: {e}")

# -------- Analyse -----------

if analyze_button:
    if not (train_spectra and measurement_spectra):
        st.error("Bitte sowohl Trainings- als auch Messdaten hochladen.")
    elif "pls_model" not in st.session_state:
        st.error("Bitte zuerst das Modell trainieren.")
    else:
        try:
            pls = st.session_state["pls_model"]
            st.subheader("Analysiere Messungen...")

            wavelengths = np.linspace(800, 200, num=601)
            X_measure = np.array([
                np.interp(wavelengths, df['Wavelength'], df['Absorbance']) for df in measurement_spectra.values()
            ])

            y_pred = pls.predict(X_measure)

            st.subheader("Messspektren")
            st.bokeh_chart(plot_spectra(measurement_spectra, "Messspektren"), use_container_width=True)

            pred_df = pd.DataFrame(y_pred, columns=components, index=list(measurement_spectra.keys()))
            pred_mean = pred_df.groupby(pred_df.index).mean()
            pred_std = pred_df.groupby(pred_df.index).std()

            st.subheader("Vorhergesagte Konzentrationen (mg/mL)")
            st.write("Mittelwert (Triplikate):")
            st.dataframe(pred_mean)
            st.write("Standardabweichung:")
            st.dataframe(pred_std)

        except Exception as e:
            st.error(f"Fehler bei der Analyse: {e}")

