import yfinance as yf 
import pandas as pd    
import numpy as np    
import streamlit as st 
import plotly.express as px
import requests

# --- DYNAMIC ECB RISK-FREE RATE FETCH ---
@st.cache_data(ttl=86400)
def get_ecb_deposit_rate():
    try:
        # Direct query to ECB SDW CSV endpoint for the Deposit Facility Rate
        ecb_url = "https://sdw-wsrest.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.DFR_LEV.LEV?lastNObservations=1&format=csvdata"
        df = pd.read_csv(ecb_url)
        if not df.empty and 'OBS_VALUE' in df.columns:
            return round(float(df['OBS_VALUE'].iloc[-1]), 2)
    except Exception:
        pass
    return 2.50  # Fallback to current 2.50% ECB rate

dynamic_rf = get_ecb_deposit_rate()

# --- SIDEBAR UI ---
st.sidebar.header("Optimiser Settings")

available_options = ["AAPL", "MSFT", "JNJ", "VTI", "BND", "TSLA", "GOOGL", "AMZN"]
selected_tickers = st.sidebar.multiselect(
    "Select assets to include:",
    options=available_options,
    default=["AAPL", "MSFT", "JNJ", "VTI", "BND"]
)

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (%) [ECB Benchmark]", 
    min_value=0.0, 
    max_value=15.0, 
    value=dynamic_rf, 
    step=0.10,
    help=f"Dynamic baseline anchored to ECB rate ({dynamic_rf:.2f}%)."
) / 100
