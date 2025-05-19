import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Whisker
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

def plot_spectra(spectra, title, wl_min, wl_max):
    p = figure(title=title, width=700, height=400, x_axis_label='Wavelength (nm)', y_axis_label='Absorbance')
    colors = Category10[10]
    for idx, (name, df) in enumerate(spectra.items()):
        df_filtered = df[(df['Wavelength'] >= wl_min) & (df['Wavelength'] <= wl_max)]
        source = ColumnDataSource(df_filtered)
        p.line('Wavelength', 'Absorbance', source=source, legend_label=name, line_width=2, color=colors[idx % len(colors)])
    p.legend.click_policy = "hide"
    return p

def calculate_sample_concentrations(stock_conc, volumes, total_volume):
    return (stock_conc.values * volumes.values) / total_volume

def clean_filename(name):
    return re.sub(r'\.csv$|\.txt$', '', name)

def plot_prediction_bars(pred_mean, pred_std, components):
    p = figure(x_range=list(pred_mean.index), title="Konzentrationen mit Fehlerbalken", height=400, width=800)
    colors = Category10[10]
    for idx, comp in enumerate(components):
        x = list(pred_mean.index)
        y = pred_mean[comp].values
        upper = y + pred_std[comp].fillna(0).values
        lower = y - pred_std[comp].fillna(0).values

        source = ColumnDataSource(data={"x": x, "y": y, "upper": upper, "lower": lower})
        p.vbar(x='x', top='y', width=0.4, source=source, color=colors[idx % len(colors)], legend_label=comp)
        p.add_layout(Whisker(source=source, base="x", upper="upper", lower="lower"))

    p.xaxis.axis_label = "Proben"
    p.yaxis.axis_label = "Konzentration (mg/mL)"
    p.legend.click_policy = "hide"
    return p

# ---------- Streamlit App ----------

st.set_page_config(page_title="UV-Vis PLS Analyse", layout="wide")

st.title("UV-Vis Spektralanalyse mit PLS")

# Sidebar Inputs
st.sidebar.header("Einstellungen")
num_components = st.sidebar.number_input("Anzahl chemischer Komponenten", min_value=1, max_value=10, value=3)
pls_components = st.sidebar.number_input("Anzahl PLS-Komponenten", min_value=1, max_value=num_components, value=num_components - 1)
wl_range = st.sidebar.slider("Spektralbereich (nm)", 200, 800, (400, 700))

uploaded_train_files = st.sidebar.file_uploader("Trainingsspektren hochladen", accept_multiple_files=True, type=['csv', 'txt'])
uploaded_measurement_files = st.sidebar.file_uploader("Messspektren hochladen", accept_multiple_files=True, type=['csv', 'txt'])

eval_button = st.sidebar.button("PLS-Komponenten evaluieren")
train_button = st.sidebar.button("Modell trainieren")
analyze_button = st.sidebar.button("Messung analysieren")

# Daten einlesen
train_spectra = {}
measurement_spectra = {}

for file in uploaded_train_files:
    df = parse_spectrum(file)
    if df is not None:
        train_spectra[clean_filename(file.name)] = df

for file in uploaded_measurement_files:
    df = parse_spectrum(file)
    if df is not None:
        measurement_spectra[clean_filename(file.name)] = df

if train_spectra:
    st.subheader("Trainingsspektren")
    st.bokeh_chart(plot_spectra(train_spectra, "Trainingsspektren", *wl_range), use_container_width=True)

    components = [f"Komponente {i+1}" for i in range(num_components)]
    st.subheader("Stammlösungs-Konzentrationen (mg/mL)")

    stock_conc_df = pd.DataFrame({"Komponente": components, "Konzentration (mg/mL)": [1.0]*num_components})
    stock_conc_df = st.data_editor(stock_conc_df, num_rows="fixed", key="stock_conc")
    stock_conc_series = pd.Series(stock_conc_df["Konzentration (mg/mL)"].values, index=components)

    st.subheader("Mischverhältnisse (Volumen in mL)")
    mixing_columns = components + ["Wasser"]
    sample_names = list(train_spectra.keys())

    if "mixing_df" not in st.session_state:
        st.session_state.mixing_df = pd.DataFrame(0.0, index=sample_names, columns=mixing_columns)
    else:
        existing = st.session_state.mixing_df
        st.session_state.mixing_df = pd.DataFrame(0.0, index=sample_names, columns=mixing_columns)
        for sample in sample_names:
            if sample in existing.index:
                st.session_state.mixing_df.loc[sample] = existing.loc[sample]

    mixing_df = st.data_editor(st.session_state.mixing_df, key="mixing_volumes")

# PLS Komponenten evaluieren
if eval_button and train_spectra:
    wavelengths = np.linspace(*wl_range[::-1], num=601)
    X = np.array([np.interp(wavelengths, df['Wavelength'], df['Absorbance']) for df in train_spectra.values()])
    y = calculate_sample_concentrations(stock_conc_series, mixing_df[components], mixing_df.sum(axis=1).values[:, np.newaxis])

    scores = []
    for n in range(1, min(num_components + 1, len(train_spectra))):
        model = PLSRegression(n_components=n)
        cv = KFold(n_splits=min(5, len(train_spectra)), shuffle=True, random_state=1)
        mse = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error').mean()
        scores.append(mse)

    st.line_chart(pd.DataFrame(scores, index=range(1, len(scores)+1), columns=['MSE']))

# Modell trainieren
if train_button and train_spectra:
    wavelengths = np.linspace(*wl_range[::-1], num=601)
    X_train = np.array([np.interp(wavelengths, df['Wavelength'], df['Absorbance']) for df in train_spectra.values()])
    y_train = calculate_sample_concentrations(stock_conc_series, mixing_df[components], mixing_df.sum(axis=1).values[:, np.newaxis])

    pls = PLSRegression(n_components=pls_components)
    pls.fit(X_train, y_train)
    st.session_state["pls_model"] = pls
    st.session_state["wavelengths"] = wavelengths
    st.success("Modell erfolgreich trainiert!")

# Analyse
if analyze_button and "pls_model" in st.session_state and measurement_spectra:
    wavelengths = st.session_state["wavelengths"]
    pls = st.session_state["pls_model"]

    X_measure = np.array([np.interp(wavelengths, df['Wavelength'], df['Absorbance']) for df in measurement_spectra.values()])
    y_pred = pls.predict(X_measure)

    st.subheader("Messspektren")
    st.bokeh_chart(plot_spectra(measurement_spectra, "Messspektren", *wl_range), use_container_width=True)

    pred_df = pd.DataFrame(y_pred, columns=components, index=list(measurement_spectra.keys()))
    pred_mean = pred_df.groupby(pred_df.index).mean()
    pred_std = pred_df.groupby(pred_df.index).std()

    st.subheader("Vorhergesagte Konzentrationen (mg/mL)")
    st.write("Mittelwert (Triplikate):")
    st.dataframe(pred_mean)
    st.write("Standardabweichung:")
    st.dataframe(pred_std)
    st.bokeh_chart(plot_prediction_bars(pred_mean, pred_std, components), use_container_width=True)
