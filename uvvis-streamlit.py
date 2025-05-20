import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Whisker
from bokeh.palettes import Category10

# -------------- Helper Functions -------------------

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
    return None

def plot_spectra(spectra, title, x_range=None):
    p = figure(title=title, width=700, height=400, x_axis_label='Wavelength (nm)', y_axis_label='Absorbance', x_range=x_range)
    colors = Category10[10]
    for idx, (name, df) in enumerate(spectra.items()):
        source = ColumnDataSource(df)
        clean_name = re.sub(r'\.csv|\.txt', '', name)
        p.line('Wavelength', 'Absorbance', source=source, legend_label=clean_name, line_width=2, color=colors[idx % len(colors)])
    p.legend.click_policy = "hide"
    return p

def calculate_sample_concentrations(stock_concentrations, volumes, total_volume):
    concentrations = (stock_concentrations.values * volumes.values) / total_volume[:, np.newaxis]
    return pd.DataFrame(concentrations, columns=stock_concentrations.index)

def evaluate_pls_components(X, y, max_components):
    mse = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X, y, cv=kf, scoring=scorer)
        mse.append((-scores).mean())
    return mse

def plot_boxplot(pred_mean, pred_std, components_names):
    p = figure(x_range=components_names, title="Vorhergesagte Konzentrationen mit Standardabweichung", width=700, height=400, y_axis_label="Konzentration (mg/mL)")
    source = ColumnDataSource(data={
        'component': components_names,
        'mean': pred_mean.values.flatten(),
        'upper': (pred_mean + pred_std).values.flatten(),
        'lower': (pred_mean - pred_std).values.flatten()
    })
    p.vbar(x='component', top='mean', width=0.6, source=source)
    whisker = Whisker(base='component', upper='upper', lower='lower', source=source)
    whisker.upper_head.size = 10
    whisker.lower_head.size = 10
    p.add_layout(whisker)
    return p


# -------------- Streamlit App -------------------

st.set_page_config(page_title="UV-Vis PLS Analyse", layout="wide")
st.title("UV-Vis Spektralanalyse mit PLS")
st.info("Trainiere ein Modell zur Vorhersage von Konzentrationen basierend auf UV-Vis Spektren.")

# --- Sidebar Inputs ---
st.sidebar.header("Einstellungen")

num_chem_components = st.sidebar.number_input("Anzahl chemischer Komponenten", min_value=1, max_value=10, value=3)

wavelength_min, wavelength_max = st.sidebar.slider("Spektralbereich (nm)", 200, 800, (200, 800))

pls_components = st.sidebar.number_input("Anzahl PLS-Komponenten (max 7)", min_value=1, max_value=7, value=3)

uploaded_train_files = st.sidebar.file_uploader("Trainingsspektren hochladen", accept_multiple_files=True, type=['csv', 'txt'])

uploaded_measurement_files = st.sidebar.file_uploader("Messspektren (Triplikate) hochladen", accept_multiple_files=True, type=['csv', 'txt'])

train_button = st.sidebar.button("Modell trainieren")
evaluate_button = st.sidebar.button("PLS evaluieren")
analyze_button = st.sidebar.button("Messung analysieren")

# --- Stammlösungskonzentrationen ---
default_stock_df = pd.DataFrame({
    "Komponente": [f"Komponente {i+1}" for i in range(num_chem_components)],
    "Konzentration (mg/mL)": [1.0]*num_chem_components
})
stock_conc_df = st.data_editor(default_stock_df, key="stock_conc_df", num_rows="fixed")

# --- Mischverhältnisse ---
mixing_columns = list(stock_conc_df["Komponente"]) + ["Solvens"]

if uploaded_train_files:
    sample_names = [file.name for file in uploaded_train_files]
    default_mixing_df = pd.DataFrame(0.0, index=sample_names, columns=mixing_columns)
    mixing_df = st.data_editor(default_mixing_df, key="mixing_df")
else:
    st.info("Bitte lade Trainingsspektren hoch, um Mischverhältnisse zu definieren.")
    mixing_df = pd.DataFrame()

x_range = (wavelength_min, wavelength_max)

# --- Trainingsspektren einlesen ---
train_spectra = {}
if uploaded_train_files:
    for file in uploaded_train_files:
        df = parse_spectrum(file)
        if df is not None:
            train_spectra[file.name] = df
        else:
            st.warning(f"{file.name} konnte nicht gelesen werden.")

# --- Messspektren einlesen ---
measurement_spectra = {}
if uploaded_measurement_files:
    for file in uploaded_measurement_files:
        df = parse_spectrum(file)
        if df is not None:
            measurement_spectra[file.name] = df
        else:
            st.warning(f"{file.name} konnte nicht gelesen werden.")

# --- Trainingsspektren plotten ---
if train_spectra:
    st.subheader("Trainingsspektren")
    st.bokeh_chart(plot_spectra(train_spectra, "Trainingsspektren", x_range=x_range), use_container_width=True)

# --- Modell Training ---
if train_button:
    if not train_spectra:
        st.error("Bitte lade Trainingsspektren hoch!")
    elif mixing_df.empty:
        st.error("Bitte gib Mischverhältnisse ein!")
    else:
        try:
            st.subheader("Modell Training...")
            wavelengths = np.linspace(wavelength_min, wavelength_max, num=601)
            X_train = np.array([
                np.interp(wavelengths, df["Wavelength"], df["Absorbance"]) for df in train_spectra.values()
            ])
            chem_components = list(stock_conc_df["Komponente"])
            stock_conc_series = pd.Series(stock_conc_df["Konzentration (mg/mL)"].values, index=chem_components)
            volumes_only = mixing_df[chem_components]
            total_volumes = volumes_only.sum(axis=1).values + mixing_df["Solvens"].values
            if not np.allclose(total_volumes, total_volumes[0], atol=0.01):
                st.error("Gesamtvolumen der Mischungen ist nicht konstant!")
            else:
                y_train = calculate_sample_concentrations(stock_conc_series, volumes_only, total_volumes)
                pls = PLSRegression(n_components=pls_components)
                pls.fit(X_train, y_train)
                st.success("Modell erfolgreich trainiert!")
                # Speichere für Analyse
                st.session_state["pls_model"] = pls
                st.session_state["wavelengths"] = wavelengths
        except Exception as e:
            st.error(f"Fehler beim Training: {e}")

# --- PLS Evaluation ---
if evaluate_button:
    if not train_spectra:
        st.error("Bitte lade Trainingsspektren hoch!")
    elif mixing_df.empty:
        st.error("Bitte gib Mischverhältnisse ein!")
    else:
        try:
            st.subheader("PLS Komponenten Evaluation (MSE)")
            wavelengths = np.linspace(wavelength_min, wavelength_max, num=601)
            X_train = np.array([
                np.interp(wavelengths, df["Wavelength"], df["Absorbance"]) for df in train_spectra.values()
            ])
            chem_components = list(stock_conc_df["Komponente"])
            stock_conc_series = pd.Series(stock_conc_df["Konzentration (mg/mL)"].values, index=chem_components)
            volumes_only = mixing_df[chem_components]
            total_volumes = volumes_only.sum(axis=1).values + mixing_df["Solvens"].values
            if not np.allclose(total_volumes, total_volumes[0], atol=0.01):
                st.error("Gesamtvolumen der Mischungen ist nicht konstant!")
            else:
                y_train = calculate_sample_concentrations(stock_conc_series, volumes_only, total_volumes)
                max_pls_eval = 7
                mse = evaluate_pls_components(X_train, y_train, max_pls_eval)
                df_mse = pd.DataFrame({"PLS-Komponenten": list(range(1, max_pls_eval+1)), "MSE": mse})
                st.line_chart(df_mse.set_index("PLS-Komponenten"))
        except Exception as e:
            st.error(f"Fehler bei der PLS Evaluation: {e}")

# --- Analyse Messspektren ---
if analyze_button:
    pls = st.session_state.get("pls_model", None)
    wavelengths = st.session_state.get("wavelengths", None)
    if pls is None or wavelengths is None:
        st.error("Bitte trainiere zuerst ein Modell!")
    elif not measurement_spectra:
        st.error("Bitte lade Messspektren hoch!")
    else:
        try:
            # Interpolation der Messdaten
            filenames = list(measurement_spectra.keys())
            X_measure = np.array([
                np.interp(wavelengths, df["Wavelength"], df["Absorbance"]) for df in measurement_spectra.values()
            ])

            # Vorhersage mit PLS-Modell
            y_pred = pls.predict(X_measure)
            chem_components = list(stock_conc_df["Komponente"])
            pred_df = pd.DataFrame(y_pred, columns=chem_components, index=filenames)

            # Berechnung von Mittelwert & empirischer Standardabweichung
            mean_row = pred_df.mean(axis=0)
            std_row = pred_df.std(axis=0, ddof=1)

            # Tabelle mit Ergebnissen anzeigen
            result_df = pred_df.copy()
            result_df.loc["Standardabweichung"] = std_row
            result_df.loc["Mittelwert"] = mean_row

            st.subheader("Vorhergesagte Konzentrationen mit Mittelwert & Standardabweichung")
            st.dataframe(result_df)

            # ---------- BALKENDIAGRAMM MIT FEHLERBALKEN ----------
            from bokeh.plotting import figure
            from bokeh.models import ColumnDataSource, Whisker

            p = figure(title="Mittelwert & Standardabweichung je Komponente",
                       x_range=chem_components,
                       width=700, height=400,
                       y_axis_label="Konzentration (mg/mL)")

            source = ColumnDataSource(data=dict(
                component=chem_components,
                mean=mean_row.values,
                upper=(mean_row + std_row).values,
                lower=(mean_row - std_row).values
            ))

            p.vbar(x='component', top='mean', width=0.6, source=source, color="steelblue")
            whisker = Whisker(base='component', upper='upper', lower='lower', source=source)
            whisker.upper_head.size = 10
            whisker.lower_head.size = 10
            p.add_layout(whisker)

            st.bokeh_chart(p, use_container_width=True)

        except Exception as e:
            st.error(f"Fehler bei der Analyse: {e}")
