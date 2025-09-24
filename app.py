import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objs as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import io
import base64

st.set_page_config(layout="wide", page_title="Latin Countries Historical Regression Explorer")

# --- Config ---
COUNTRIES = {
    "Chile": "CHL",
    "Uruguay": "URY",
    "Panama": "PAN"
}

# Indicator mapping (World Bank codes and some OWID sources)
INDICATORS = {
    "Population": {"source": "worldbank", "code": "SP.POP.TOTL", "units": "people"},
    "Unemployment rate": {"source": "worldbank", "code": "SL.UEM.TOTL.ZS", "units": "% of labor force"},
    "Education (0-25 scale, mean years -> scaled)": {"source": "owid", "code": "average-years-of-schooling", "units": "0-25 scale (mapped)"},
    "Life expectancy": {"source": "worldbank", "code": "SP.DYN.LE00.IN", "units": "years"},
    "Average income (GDP per capita, current US$)": {"source": "worldbank", "code": "NY.GDP.PCAP.CD", "units": "current US$ (per person)"},
    "Birth rate": {"source": "worldbank", "code": "SP.DYN.CBRT.IN", "units": "births per 1,000 people"},
    "Net migration": {"source": "worldbank", "code": "SM.POP.NETM", "units": "net migrants"},
    "Murder Rate": {"source": "worldbank", "code": "VC.IHR.PSRC.P5", "units": "intentional homicides per 100,000 people"}
}

MIN_YEAR = 1950
MAX_YEAR = datetime.now().year

st.title("Latin Countries — 70‑Year Historical Regression Explorer")
st.markdown("""
This app fetches real historical indicators (World Bank and Our World in Data) for up to three of the wealthiest Latin countries (Chile, Uruguay, Panama),
fits a polynomial regression (degree ≥ 3), and performs function analysis (local extrema, increasing/decreasing, fastest change, extrapolation, etc.).
Notes:
- Education is taken from Our World in Data (average years of schooling) and scaled to a 0–25 scale for readability (see app explanations).
- If an indicator lacks long data going back 70 years, the app uses the available reliable years and tells you the covered span.
""")

with st.expander("Which countries are included & why?"):
    st.write("This demo uses Chile, Uruguay, and Panama as examples (they are among the highest GDP-per-capita Latin American countries in recent decades).")
    st.write("If you want other countries, paste a CSV or request them and I'll adapt the app.")

# Sidebar controls
st.sidebar.header("Controls")
indicator = st.sidebar.selectbox("Choose indicator (category)", list(INDICATORS.keys()))
countries_selected = st.sidebar.multiselect("Select countries to include (compare on same chart)", list(COUNTRIES.keys()), default=list(COUNTRIES.keys())[:1])
degree = st.sidebar.number_input("Polynomial degree (use 3 or higher)", min_value=3, max_value=10, value=3, step=1)
year_step = st.sidebar.slider("Plot points every N years (sampling for visibility)", 1, 10, 1)
extrapolate_years = st.sidebar.slider("Extrapolate forward this many years (0 = none)", 0, 50, 5)
show_extrapolated = st.sidebar.checkbox("Highlight extrapolated portion", value=True)
download_data = st.sidebar.button("Download raw data (CSV)")

st.sidebar.markdown("**Other tools**")
interp_year = st.sidebar.number_input("Year to interpolate/extrapolate value for:", min_value=MIN_YEAR, max_value=MAX_YEAR+50, value=MAX_YEAR+1)
roc_y1 = st.sidebar.number_input("Average rate of change — start year", min_value=MIN_YEAR, max_value=MAX_YEAR+50, value=2000)
roc_y2 = st.sidebar.number_input("Average rate of change — end year", min_value=MIN_YEAR, max_value=MAX_YEAR+50, value=2020)
print_friendly = st.sidebar.button("Generate printer-friendly report (HTML)")

# Option to upload CSV for US Latin groups comparison
st.sidebar.markdown("---")
st.sidebar.markdown("Compare Latin groups in the U.S.: upload CSV with columns 'group', 'year', 'value'")
us_groups_file = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

# Helper: fetch World Bank data
@st.cache_data(ttl=3600)
def fetch_wb(country_code, indicator_code, year_from=MIN_YEAR, year_to=MAX_YEAR):
    # World Bank API uses years mostly from 1960 onward, but we'll request broadly and then filter
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={year_from}:{year_to}&per_page=20000&format=json"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    if not isinstance(data, list) or len(data) < 2:
        return pd.DataFrame()
    rows = data[1]
    records = []
    for item in rows:
        year = item.get("date")
        val = item.get("value")
        if val is not None:
            records.append({"year": int(year), "value": float(val)})
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    return df

@st.cache_data(ttl=3600)
def fetch_owid(series_name):
    # Our World in Data graphers expose CSV endpoints predictable for some series
    base = "https://ourworldindata.org/grapher/"
    url = f"{base}{series_name}.csv"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(r.text))
    return df

# Load and prepare data for selected countries
data_tables = {}
min_year_available = MAX_YEAR
max_year_available = MIN_YEAR

for c in countries_selected:
    code = COUNTRIES[c]
    meta = INDICATORS[indicator]
    if meta["source"] == "worldbank":
        df = fetch_wb(code, meta["code"], year_from=MIN_YEAR, year_to=MAX_YEAR)
        if df.empty:
            st.warning(f"No World Bank data found for {c} / {indicator}.")
            continue
        data_tables[c] = df
        min_year_available = min(min_year_available, int(df.year.min()))
        max_year_available = max(max_year_available, int(df.year.max()))
    else:  # owid
        df_owid = fetch_owid(meta["code"])
        if df_owid.empty:
            st.warning(f"No OWID data found for {indicator}.")
            continue
        # OWID format: columns for countries
        if c in df_owid.columns:
            df = df_owid[["Year", c]].rename(columns={"Year":"year", c:"value"})
            df = df.dropna().astype({"year":int, "value":float}).sort_values("year")
            data_tables[c] = df
            min_year_available = min(min_year_available, int(df.year.min()))
            max_year_available = max(max_year_available, int(df.year.max()))
        else:
            st.warning(f"{c} not in OWID series {meta['code']}")

if not data_tables:
    st.stop()

# Show data range used
st.sidebar.markdown(f"Data span used across selected countries: **{min_year_available} — {max_year_available}** (available years)")

# Display raw editable table for primary country (first selected)
primary = countries_selected[0] if countries_selected else list(data_tables.keys())[0]
st.subheader(f"Raw data (editable) — {indicator} — {primary}")
df_primary = data_tables[primary].copy()
df_primary = df_primary.set_index("year")
edited = st.experimental_data_editor(df_primary, num_rows="dynamic")
# allow user to download edited CSV
csv_buffer = edited.reset_index().to_csv(index=False).encode("utf-8")
st.download_button("Download edited data (CSV)", csv_buffer, file_name=f"{primary}_{indicator.replace(' ','_')}_data.csv", mime="text/csv")

# Function to fit polynomial regression and return model and x-range
def fit_poly_model(years, values, degree):
    # years: numpy array (1D)
    X = years.reshape(-1,1)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = poly.fit_transform(X)
    model = LinearRegression().fit(Xp, values)
    # construct coefficients in decreasing powers for numpy.poly1d
    # we can build polynomial coefficients by evaluating over basis
    coefs = np.zeros(degree+1)
    # compute polynomial coefficients by fitting powers explicitly
    # Solve for coefficients a0 + a1*x + a2*x^2 ...
    # Use Vandermonde
    V = np.vander(years, N=degree+1)
    a, *_ = np.linalg.lstsq(V, values, rcond=None)
    return np.poly1d(a), model, poly

# Plotting
fig = go.Figure()
colors = ["blue", "green", "red", "orange", "purple"]

for i, (country, df) in enumerate(data_tables.items()):
    years = df['year'].values
    vals = df['value'].values
    # sample according to year_step
    mask = ((years - years.min()) % year_step == 0) if year_step>1 else np.ones_like(years,dtype=bool)
    years_plot = years[mask]
    vals_plot = vals[mask]
    # Fit polynomial (use year as x; to improve numerical stability, shift years by subtracting mean)
    x = years.astype(float)
    # shift to reduce conditioning issues
    x_mean = x.mean()
    x_shift = x - x_mean
    p, model, poly = fit_poly_model(x_shift, vals, degree)
    # Create dense x for curve within data range
    x_dense = np.linspace(x_shift.min(), x_shift.max(), 400)
    y_fit = p(x_dense)
    # Extrapolation if requested
    if extrapolate_years>0:
        last = x_shift.max()
        x_ex = np.linspace(last, last + extrapolate_years, 200)
        y_ex = p(x_ex)
    else:
        x_ex = np.array([]); y_ex = np.array([])
    # Plot scatter
    fig.add_trace(go.Scatter(x=years, y=vals, mode="markers", name=f"{country} (data)", marker=dict(color=colors[i%len(colors)])))
    # Plot fitted curve (within data)
    fig.add_trace(go.Scatter(x=(x_dense + x_mean), y=y_fit, mode="lines", name=f"{country} fit", line=dict(color=colors[i%len(colors)], width=2)))
    # Extrapolated portion
    if x_ex.size>0:
        fig.add_trace(go.Scatter(x=(x_ex + x_mean), y=y_ex, mode="lines", name=f"{country} extrapolation", line=dict(color=colors[i%len(colors)], width=2, dash="dash")))

fig.update_layout(title=f"{indicator} — data and polynomial fits (degree {degree})", xaxis_title="Year", yaxis_title=INDICATORS[indicator]['units'], legend=dict(orientation="h"))

st.plotly_chart(fig, use_container_width=True)

# Display equations and function analysis for primary country
st.subheader("Model equation & function analysis (primary country)")
df = data_tables[primary]
x = df['year'].values.astype(float)
y = df['value'].values.astype(float)
x_mean = x.mean()
x_shift = x - x_mean
poly_obj, _, _ = fit_poly_model(x_shift, y, degree)
# show polynomial coefficients in human-readable form (in terms of year)
coef = poly_obj.coeffs  # highest-first
terms = []
deg = len(coef)-1
for idx, c in enumerate(coef):
    power = deg - idx
    if abs(c) < 1e-12:
        continue
    term = f"{c:.6g}* (year - {x_mean:.1f})^{power}" if power>0 else f"{c:.6g}"
    terms.append(term)
equation = " + ".join(terms)
st.markdown(f"**Polynomial (in shifted year = year - {x_mean:.1f}):**\n\n`f(year) = {equation}`")
# Provide nicer form: show as poly1d with absolute years
st.markdown("**Readable polynomial (expanded for year variable)**")
coeffs_year = np.poly1d(poly_obj).coeffs  # still shifted poly, so we won't expand further here for brevity
st.latex(str(poly_obj))

# Function analysis: derivatives
p = poly_obj
p1 = np.polyder(p)
p2 = np.polyder(p1)
# Find critical points in shifted x domain
roots = np.roots(p1)
real_roots = sorted([r.real for r in roots if abs(r.imag) < 1e-6 and x_shift.min()-5 <= r.real <= x_shift.max()+5])
analysis_text = []
if real_roots:
    for r in real_roots:
        yr = r + x_mean
        val = p(r)
        second = p2(r)
        kind = "local minimum" if second>0 else ("local maximum" if second<0 else "inflection (flat)") 
        analysis_text.append(f"The {indicator.lower()} for {primary} reached a **{kind}** on year **{int(round(yr))}** with value **{val:.3f} {INDICATORS[indicator]['units']}**.")
else:
    analysis_text.append("No real critical points (local extrema) found inside/near the data range for this polynomial approximation.")

# Increasing/decreasing intervals (sample)
xs = np.linspace(x_shift.min()-5, x_shift.max()+5, 500)
deriv_vals = p1(xs)
incs = []
starts = None
for xi, dv in zip(xs, deriv_vals):
    if dv>0 and starts is None:
        starts = xi
    if dv<=0 and starts is not None:
        incs.append((starts + x_mean, xi + x_mean))
        starts = None
if starts is not None:
    incs.append((starts + x_mean, xs[-1] + x_mean))
if incs:
    analysis_text.append("The function is increasing approximately on intervals (years): " + ", ".join([f"{int(a)} to {int(b)}" for a,b in incs]) + ".")
else:
    analysis_text.append("No intervals of sustained increase found by sampling derivative sign.")

# Fastest increasing/decreasing: max absolute derivative
abs_d = np.abs(deriv_vals)
idx_max = np.argmax(abs_d)
fast_x = xs[idx_max] + x_mean
fast_rate = deriv_vals[idx_max]
analysis_text.append(f"The function is changing fastest (largest |df/dt|) around year {int(round(fast_x))} at a rate of {fast_rate:.4g} {INDICATORS[indicator]['units']}/year.")

# Domain & range (practical)
domain_text = f"Domain used (data years): {int(x.min())} to {int(x.max())}. Polynomial defined for all real years, but predictions outside data range are extrapolations."
range_est = (p(xs).min(), p(xs).max())
analysis_text.append(f"Empirical range (within sampled years): approximately {range_est[0]:.3f} to {range_est[1]:.3f} {INDICATORS[indicator]['units']}.")

for line in analysis_text:
    st.write(line)

# Extrapolation prediction example
pred_year = interp_year
shifted = pred_year - x_mean
pred_val = p(shifted)
is_extrap = (pred_year < x.min()) or (pred_year > x.max())
st.markdown(f"### Prediction for year {pred_year} — {'extrapolation' if is_extrap else 'interpolation'}")
st.write(f"Predicted {indicator.lower()} for {primary} in {pred_year}: **{pred_val:.3f} {INDICATORS[indicator]['units']}**.")

# Average rate of change between two years via model
y1 = p(roc_y1 - x_mean)
y2 = p(roc_y2 - x_mean)
avg_roc = (y2 - y1) / (roc_y2 - roc_y1) if roc_y2!=roc_y1 else np.nan
st.markdown("### Average rate of change (model)")
st.write(f"From {roc_y1} to {roc_y2}, the model predicts a change from {y1:.3f} to {y2:.3f} {INDICATORS[indicator]['units']}, average rate {avg_roc:.4g} {INDICATORS[indicator]['units']}/year.")

# Conjectures (simple automatic suggestions based on years of big changes)
st.subheader("Conjectures about significant changes (automated hints)")
# find biggest year-to-year jumps in original data
if len(x)>1:
    diffs = np.abs(np.diff(y))
    idx = np.argmax(diffs)
    year_a = int(x[idx]); year_b = int(x[idx+1])
    st.write(f"The largest year-to-year absolute change in the raw data occurred between {year_a} and {year_b} (change of {diffs[idx]:.3f} {INDICATORS[indicator]['units']}). Consider investigating historical events (economic crises, policy changes, conflicts, epidemics) around those years.")
else:
    st.write("Not enough raw data to highlight large changes.")

# Allow multiple countries overlay export
st.markdown("### Multi-country download & print")
if download_data:
    # combine all data into zip-like CSV stream
    combined = []
    for country, dfc in data_tables.items():
        tmp = dfc.copy(); tmp['country']=country; combined.append(tmp)
    all_df = pd.concat(combined)
    st.download_button("Download combined CSV", all_df.to_csv(index=False).encode('utf-8'), file_name="combined_data.csv", mime="text/csv")

# Printer-friendly report generation (simple HTML)
if print_friendly:
    html = f\"\"\"
    <html><head><meta charset='utf-8'><title>Printer-friendly report</title></head><body>
    <h1>{indicator} — {primary}</h1>
    <h2>Data span: {int(x.min())} — {int(x.max())}</h2>
    <h3>Model equation (shifted year)</h3><pre>{equation}</pre>
    <h3>Predicted {pred_year}: {pred_val:.3f} {INDICATORS[indicator]['units']}</h3>
    <h3>Function analysis</h3>
    <ul>
    \"\"\"
    for line in analysis_text:
        html += f"<li>{line}</li>"
    html += "</ul><h3>Raw data</h3>"
    html += edited.to_html(index=False)
    html += "</body></html>"
    b = html.encode('utf-8')
    st.download_button("Download printer-friendly HTML", b, file_name=f"report_{primary}_{indicator.replace(' ','_')}.html", mime="text/html")

st.markdown("---")
st.caption("Sources: World Bank API (https://data.worldbank.org), Our World in Data (https://ourworldindata.org). This app fits polynomials for demonstration and educational analysis; real forecasting should use domain-specific time series models and careful validation.")
