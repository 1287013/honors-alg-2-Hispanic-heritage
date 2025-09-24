import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import plotly.graph_objs as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import io

st.set_page_config(layout="wide", page_title="Latin Countries Historical Regression Explorer")

# --- Config ---
COUNTRIES = {
    "Chile": "CHL",
    "Uruguay": "URY",
    "Panama": "PAN"
}

INDICATORS = {
    "Population": {"source": "worldbank", "code": "SP.POP.TOTL", "u
