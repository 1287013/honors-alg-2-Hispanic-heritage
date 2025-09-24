
"""
Streamlit app: Historical data regression & analysis for up to 3 Latin countries

How it works (short):
 - Pulls historical indicators from the World Bank API (and Our World In Data for mean years of schooling).
 - Lets you select an indicator (Population, Life expectancy, Birth rate, GDP per capita as average income/wealth,
   Unemployment, Net migration (immigration out), Homicide rate).
 - Fits a polynomial regression (degree 3+; default 3) to the chosen country's data (uses year as x).
 - Shows editable table of the raw data, scatter plot + fitted curve, equation, function analysis (derivative),
   extrapolation, interpolation, average rate of change, and printable "printer-friendly" view.
 - Compare multiple countries on same plot and compare US Latin groups (placeholder: uses same country series
   because US subgroup data isn't provided by World Bank — the app allows manual table edits for custom groups).

Notes:
 - The app dynamically downloads data from World Bank. Coverage depends on indicator; many WDI indicators begin in 1960.
 - If you want guaranteed 70-year coverage but the indicator doesn't have it, edit the table manually (the editor is enabled).
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import io
import base64
import json
import math
import streamlit.components.v1 as components

st.set_page_config(page_title="Latin Countries — Historical Regression Explorer", layout="wide")

# --- Helper functions ---

WORLD_BANK_BASE = "https://api.worldbank.org/v2/country/{}/indicator/{}?date={}&format=json&per_page=20000"

INDICATOR_MAP = {
    "Population": ("SP.POP.TOTL", "Population, total (midyear estimate) — people"),
    "Life expectancy": ("SP.DYN.LE00.IN", "Life expectancy at birth, total — years"),
    "Birth rate": ("SP.DYN.CBRT.IN", "Crude birth rate (per 1,000 people)"),
    "Average income": ("NY.GDP.PCAP.CD", "GDP per capita (current US$) — US$"),
    "Unemployment rate": ("SL.UEM.TOTL.ZS", "Unemployment, total (% of labor force) — pct"),
    "Immigration (net migration)": ("SM.POP.NETM", "Net migration — number of people"),
    "Murder Rate": ("VC.IHR.PSRC.P5", "Intentional homicides (per 100,000 people)"),
    # Education will use Our World in Data endpoint (mean years of schooling) or allow manual editing:
    "Education levels (0-25)": ("EDU_MEAN_YEARS", "Mean years of schooling (converted to 0-25 scale)")
}

# Countries: up to 3 wealthiest Latin countries (user-specified); defaults chosen for long coverage.
# We choose Argentina, Chile, Uruguay as examples with long World Bank coverage.
COUNTRY_CODE_MAP = {"Argentina":"ARG", "Chile":"CHL", "Uruguay":"URY", "Mexico":"MEX", "Panama":"PAN"}

@st.cache_data(show_spinner=False)
def fetch_worldbank(country_code, indicator_code, start_year):
    # Fetch data from World Bank from start_year to current year
    end_year = datetime.now().year
    date_range = f"{start_year}:{end_year}"
    url = WORLD_BANK_BASE.format(country_code.lower(), indicator_code, date_range)
    r = requests.get(url)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    # data[1] contains list of year-value dicts if available
    if not isinstance(data, list) or len(data) < 2 or data[1] is None:
        return None
    records = []
    for entry in data[1]:
        year = entry.get("date")
        value = entry.get("value")
        if year is None:
            continue
        records.append({"year": int(year), "value": value})
    if not records:
        return None
    df = pd.DataFrame(records).dropna()
    df = df.sort_values("year").reset_index(drop=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def fetch_owid_mean_years(country_name):
    # Our World In Data provides mean years of schooling series (ourworldindata.org)
    # We'll use the GitHub CSV (no external heavy parsing). Here we attempt to fetch a common URL pattern.
    # If fetch fails, return None.
    base = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Mean%20years%20of%20schooling%20-%20Barro%20&%20Lee%20(2015)/Mean%20years%20of%20schooling%20-%20Barro%20&%20Lee%20(2015).csv"
    try:
        r = requests.get(base, timeout=10)
        if r.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(r.text))
        # The dataset contains country column and year columns
        if "country" in df.columns:
            row = df[df["country"]==country_name]
            if row.empty:
                return None
            # melt year columns
            yrs = [c for c in df.columns if c.isdigit()]
            s = row[yrs].T
            s.index = s.index.astype(int)
            s = s.reset_index()
            s.columns = ["year","value"]
            s = s.dropna().sort_values("year").reset_index(drop=True)
            return s
        return None
    except Exception:
        return None

def poly_fit_and_predict(years, values, degree, x_predict):
    # Fit polynomial (least squares) and return coefficients (highest-first), predictions
    # We will use numpy.polyfit for convenience; for numerical stability rescale years to year - mean
    x = np.array(years, dtype=float)
    y = np.array(values, dtype=float)
    x_mean = x.mean()
    x_scaled = x - x_mean
    coeffs = np.polyfit(x_scaled, y, degree)  # highest-first
    p = np.poly1d(coeffs)
    # predict for x_predict years (years array)
    x_pred = np.array(x_predict, dtype=float)
    y_pred = p(x_pred - x_mean)
    return coeffs, p, x_mean, y_pred

def poly_derivative(poly):
    return np.polyder(poly)

def poly_integral(poly):
    return np.polyint(poly)

def poly_to_equation_text(coeffs, x_mean):
    # coeffs highest first
    deg = len(coeffs)-1
    parts = []
    for i,c in enumerate(coeffs):
        pwr = deg - i
        # display coefficient with 6 sig figs
        if abs(c) < 1e-12:
            continue
        coeff_str = f"{c:.6g}"
        if pwr == 0:
            parts.append(f"{coeff_str}")
        elif pwr == 1:
            parts.append(f"{coeff_str}*(x-{x_mean:.1f})")
        else:
            parts.append(f"{coeff_str}*(x-{x_mean:.1f})**{pwr}")
    eq = " + ".join(parts)
    return "f(x) = " + eq

def make_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# --- UI ---

st.title("Latin Countries — Historical Regression Explorer")
st.markdown("Load historical World Bank data, fit a polynomial (degree ≥ 3), analyze and extrapolate.")

# Country selection (allow up to 3)
st.sidebar.header("Data & countries")
chosen_countries = st.sidebar.multiselect("Select up to 3 countries (Latin)", list(COUNTRY_CODE_MAP.keys()),
                                          default=["Chile","Argentina","Uruguay"], help="Pick up to 3")
if len(chosen_countries) == 0:
    st.sidebar.warning("Pick at least one country.")
    st.stop()
if len(chosen_countries) > 3:
    st.sidebar.error("Please select at most 3 countries.")
    st.stop()

indicator = st.sidebar.selectbox("Select data category", list(INDICATOR_MAP.keys()), index=0)
start_year = st.sidebar.number_input("Earliest year to request from data source (min 1950)", value=1950, min_value=1900, max_value=datetime.now().year)
degree = st.sidebar.number_input("Polynomial regression degree (≥3)", min_value=3, max_value=8, value=3, step=1)

# Granularity for plotting (1-10 year increments)
x_increment = st.sidebar.slider("Plot Year Increments (years)", 1, 10, 1)

# Extrapolation years
extrap_years = st.sidebar.number_input("Extrapolate forward this many years (0 = none)", min_value=0, max_value=100, value=10)

compare_multiple = st.sidebar.checkbox("Show multiple countries on same graph (compare)")

# Option to compare Latin groups in U.S. (placeholder: allows manual table input)
compare_us_groups = st.sidebar.checkbox("Compare Latin groups living in the U.S. (manual input)")

st.sidebar.markdown("---")
if st.sidebar.button("Printer-friendly view"):
    st.session_state["print_view"] = not st.session_state.get("print_view", False)

# Fetch & prepare data per country
all_country_dfs = {}
for c in chosen_countries:
    code = COUNTRY_CODE_MAP[c]
    if indicator == "Education levels (0-25)":
        df = fetch_owid_mean_years(c)
        if df is not None:
            # Map mean years to 0-25 scale by clamping to 25 maximum
            df["value"] = df["value"].astype(float).clip(lower=0, upper=25)
    else:
        ind_code = INDICATOR_MAP[indicator][0]
        df = fetch_worldbank(code, ind_code, start_year)
    if df is None or df.empty:
        st.warning(f"No data found for {c} — indicator may not be available or coverage is limited.")
        # create an empty dataframe placeholder so user can edit
        df = pd.DataFrame({"year": list(range(start_year, datetime.now().year+1, 1)), "value": [np.nan]* (datetime.now().year - start_year +1)})
    all_country_dfs[c] = df

# Show editable tables
st.subheader("Raw data (editable)")
cols = st.columns(len(chosen_countries))
editable_tables = {}
for i, c in enumerate(chosen_countries):
    with cols[i]:
        st.markdown(f"**{c} — {indicator}**")
        df = all_country_dfs[c].copy()
        # place a data editor (Streamlit >= 1.18 has st.data_editor)
        try:
            edited = st.data_editor(df, num_rows="dynamic", key=f"editor_{c}")
        except Exception:
            # fallback to a simple table if data_editor is unavailable
            edited = st.experimental_data_editor(df, num_rows="dynamic", key=f"editor_fallback_{c}")
        editable_tables[c] = edited

# Option to download edited table csvs
for c, df in editable_tables.items():
    st.markdown(f"{make_download_link(df, filename=f'{c}_{indicator}.csv')}", unsafe_allow_html=True)

# Main analysis area
st.subheader("Analysis & Regression")

plot_fig = go.Figure()
legend_entries = []
analysis_texts = []

for c, df in editable_tables.items():
    # drop rows without numeric values
    df2 = df.copy()
    df2 = df2.dropna(subset=["year","value"])
    df2["year"] = pd.to_numeric(df2["year"], errors="coerce")
    df2 = df2.dropna(subset=["year","value"])
    if df2.empty or len(df2) < degree+1:
        st.warning(f"Insufficient numeric data for {c} to fit a degree {degree} polynomial. Need at least {degree+1} points.")
        continue
    years = df2["year"].astype(int).to_numpy()
    values = df2["value"].astype(float).to_numpy()
    # Fit polynomial
    coeffs, poly, x_mean, _ = poly_fit_and_predict(years, values, degree, years)
    eq_text = poly_to_equation_text(coeffs, x_mean)
    # Prepare x for plotting: from min(years) to max(years)+extrap_years with step x_increment
    x_min = int(years.min())
    x_max = int(years.max()) + int(extrap_years)
    x_plot = np.arange(x_min, x_max+1, x_increment)
    _, _, _, y_plot = poly_fit_and_predict(years, values, degree, x_plot)
    # separate extrapolated region
    x_future_mask = x_plot > years.max()
    # Plot original scatter
    plot_fig.add_trace(go.Scatter(x=years, y=values, mode="markers", name=f"{c} data", marker=dict(symbol="circle"), opacity=0.8))
    # Plot fitted curve (past)
    plot_fig.add_trace(go.Scatter(x=x_plot[~x_future_mask], y=y_plot[~x_future_mask], mode="lines", name=f"{c} fit", line=dict(dash="solid")))
    # Plot extrapolation (future) in dashed/different color
    if extrap_years > 0:
        plot_fig.add_trace(go.Scatter(x=x_plot[x_future_mask], y=y_plot[x_future_mask], mode="lines", name=f"{c} extrapolation", line=dict(dash="dot")))
    # Function analysis
    # derivative polynomial
    deriv = poly_derivative(poly)
    # second derivative
    second_deriv = poly_derivative(deriv)
    # find critical points (roots of derivative) within domain (we'll check within [min,max+extrap])
    try:
        crit_points = np.roots(deriv)
        crit_real = [float(r) for r in crit_points if abs(np.imag(r)) < 1e-6]
    except Exception:
        crit_real = []
    crit_in_domain = [r + x_mean for r in crit_real if (r + x_mean) >= x_min and (r + x_mean) <= x_max]
    # Evaluate maxima/minima
    extrema_texts = []
    for cp in crit_in_domain:
        # evaluate second derivative at that point
        sd_val = second_deriv(cp - x_mean)
        pt_val = poly(cp - x_mean)
        kind = "minimum" if sd_val > 0 else ("maximum" if sd_val < 0 else "inflection")
        extrema_texts.append((cp, pt_val, kind))
    # fastest increase/decrease: where derivative magnitude is largest within domain: sample fine grid
    sample_x = np.linspace(x_min, x_max, 1000)
    deriv_vals = deriv(sample_x - x_mean)
    idx_max = np.argmax(deriv_vals)
    idx_min = np.argmin(deriv_vals)
    fastest_inc_x = sample_x[idx_max]
    fastest_dec_x = sample_x[idx_min]
    fastest_inc_rate = deriv_vals[idx_max]
    fastest_dec_rate = deriv_vals[idx_min]
    # domain & range (based on polynom, but we'll show practical domain = [min year, max year+extrap])
    domain_txt = f"[{x_min}, {x_max}] (years)"
    range_vals = y_plot
    range_txt = f"[{np.nanmin(range_vals):.3g}, {np.nanmax(range_vals):.3g}] (units per indicator)"
    # Build conversational texts
    single_text = []
    single_text.append(f"**{c} — regression degree {degree}**")
    single_text.append(eq_text)
    if extrema_texts:
        for cp, val, kind in extrema_texts:
            single_text.append(f"The {indicator.lower()} of {c} reached a local {kind} on year {int(round(cp))}. The {indicator.lower()} was about {val:.3g} at that time.")
    else:
        single_text.append("No local extrema detected inside the domain.")
    single_text.append(f"The function is defined (for our analysis) on the domain {domain_txt}. Range (estimated for that domain): {range_txt}.")
    single_text.append(f"The {indicator.lower()} was increasing at its fastest rate on ~{int(round(fastest_inc_x))} at an estimated rate of {fastest_inc_rate:.3g} (units per year).")
    single_text.append(f"The {indicator.lower()} was decreasing at its fastest rate on ~{int(round(fastest_dec_x))} at an estimated rate of {fastest_dec_rate:.3g} (units per year).")
    # Extrapolation example: predict value at x = max_year + extrap_years
    if extrap_years > 0:
        target_year = years.max() + extrap_years
        _,_,_, pred = poly_fit_and_predict(years, values, degree, [target_year])
        single_text.append(f"According to the regression model, the {indicator.lower()} is predicted to be {pred[0]:.3g} in the year {int(target_year)} (this is an extrapolation and has greater uncertainty).")
    # Add to lists
    analysis_texts.append("\n\n".join(single_text))

# Layout plotting
plot_fig.update_layout(title=f"{indicator} — data & polynomial fit", xaxis_title="Year", yaxis_title=INDICATOR_MAP.get(indicator,(None,""))[1] if indicator in INDICATOR_MAP else "", legend=dict(orientation="h"))
st.plotly_chart(plot_fig, use_container_width=True)

st.markdown("### Interpretation & function analysis")
for t in analysis_texts:
    st.markdown(t)

# Interpolation & extrapolation utility
st.subheader("Interpolation / Extrapolation & Rate of Change tools")
st.markdown("Pick a country and an input year to interpolate or extrapolate from the fitted polynomial model (degree chosen earlier).")

selected_country_tool = st.selectbox("Country for prediction", chosen_countries)
year_input = st.number_input("Year to predict (can be outside data range)", min_value=1900, max_value=2100, value=int(datetime.now().year)+1)
if selected_country_tool:
    df_tool = editable_tables[selected_country_tool].dropna(subset=["year","value"])
    if len(df_tool) >= degree+1:
        yrs = df_tool["year"].astype(int).to_numpy()
        vals = df_tool["value"].astype(float).to_numpy()
        coeffs, poly, x_mean, _ = poly_fit_and_predict(yrs, vals, degree, yrs)
        pred = poly(year_input - x_mean)
        st.write(f"Predicted {indicator} for {selected_country_tool} in year {year_input}: **{pred:.3g}** (units: see indicator).")
    else:
        st.warning("Insufficient data to make prediction for this country.")

# Average rate of change between two years
st.markdown("#### Average rate of change (based on model)")
c_roc = st.selectbox("Country for average rate of change", chosen_countries, key="roc_country")
y1 = st.number_input("Start year", min_value=1900, max_value=2100, value=2000, key="roc_y1")
y2 = st.number_input("End year", min_value=1900, max_value=2100, value=2020, key="roc_y2")
if y2 <= y1:
    st.info("Pick end year greater than start year.")
else:
    df_roc = editable_tables[c_roc].dropna(subset=["year","value"])
    if len(df_roc) >= degree+1:
        yrs = df_roc["year"].astype(int).to_numpy()
        vals = df_roc["value"].astype(float).to_numpy()
        coeffs, poly, x_mean, _ = poly_fit_and_predict(yrs, vals, degree, yrs)
        v1 = poly(y1 - x_mean)
        v2 = poly(y2 - x_mean)
        avg_rate = (v2 - v1) / (y2 - y1)
        st.write(f"Average rate of change of {indicator.lower()} for {c_roc} from {y1} to {y2}: **{avg_rate:.6g} (units/year)**. ({v1:.3g} → {v2:.3g})")
    else:
        st.warning("Insufficient data to compute average rate of change.")

# Printer-friendly view using a simple HTML + print button
if st.session_state.get("print_view", False):
    st.markdown("---")
    st.markdown("### Printer-friendly summary")
    for c,t in zip(chosen_countries, analysis_texts):
        st.markdown(f"#### {c}")
        st.markdown(t)
    # print button using JS
    components.html("<script>function p(){window.print();}</script><button onclick='p()'>Print this page</button>", height=80)

st.markdown("---")
st.markdown("**Notes & Data sources:** Data is fetched from the World Bank API (https://data.worldbank.org) and Our World in Data where indicated. Model is polynomial regression (least-squares). Extrapolations beyond observed data are uncertain and should be used cautiously.")
