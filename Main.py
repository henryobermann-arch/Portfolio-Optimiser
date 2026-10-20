import yfinance as yf 
import pandas as pd    
import numpy as np    
import streamlit as st 
import plotly.express as px
from scipy.optimize import minimize

# --- FIXED SETTINGS ---
START_DATE = "2023-10-20"   # fixed window -> same data every run
END_DATE = "2026-10-20"
SEED = 42                   # fixed seed -> same random portfolios every run
FALLBACK_RF = 0.025

# --- STEP 1: INITIAL DATA SETUP ---
assets = [
    {"ticker": "AAPL", "expected_return": 0.10, "risk_score": 5},
        {"ticker": "MSFT", "expected_return": 0.12, "risk_score": 4},
            {"ticker": "JNJ",  "expected_return": 0.05, "risk_score": 2},
                {"ticker": "VTI",  "expected_return": 0.08, "risk_score": 3},
                    {"ticker": "BND",  "expected_return": 0.03, "risk_score": 1}
]

# --- STEP 2: PORTFOLIO WEIGHTS ---
# Order matches: AAPL, MSFT, JNJ, VTI, BND
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# --- STEP 3: CALCULATING PORTFOLIO RETURN ---
portfolio_return = 0
for i, asset in enumerate(assets):
    contribution = asset["expected_return"] * weights[i]
    portfolio_return += contribution

# --- STEP 4: CALCULATING PORTFOLIO RISK ---
portfolio_risk = 0
for i, asset in enumerate(assets):
    portfolio_risk += asset["risk_score"] * weights[i]

# --- PHASE 2: REAL DATA INTEGRATION ---
ticker_list = [asset["ticker"] for asset in assets]
print(f"Fetching 3 years of data for: {ticker_list}")

raw_data = yf.download(ticker_list, start=START_DATE, end=END_DATE, auto_adjust=True)

if raw_data.empty:
    st.error("Error: No data found.")
    st.stop()

print("Successfully fetched real price data!")

# --- PHASE 2, TASK 2: REAL RETURNS CALCULATION ---
close_prices = raw_data["Close"][ticker_list]
daily_returns = close_prices.pct_change()
annual_returns = daily_returns.mean() * 252

print("Annualised expected returns based on 3y history:")
print(annual_returns)

# --- PHASE 2, TASK 3: REAL VOLATILITY CALCULATION ---
daily_returns = close_prices.pct_change()
annual_volatility = daily_returns.std() * np.sqrt(252)

print("Annualised volatility (real risk):")
print(annual_volatility)

# --- PHASE 2: STEP 4, OVERWRITING WITH REAL DATA ---
for asset in assets:
    ticker_name = asset["ticker"] 
    asset["expected_return"] = annual_returns[ticker_name] 
    asset["risk_score"] = annual_volatility[ticker_name] 

print("Successfully integrated real market data into the asset list.")
print(f"New data for {assets[0]['ticker']}: Return={assets[0]['expected_return']:.2%}, Risk={assets[0]['risk_score']:.2%}")

# --- PHASE 2: STEP 5, ERROR HANDLING & ECB RATE ---
@st.cache_data(ttl=86400)
def get_real_data(ticker_list):
    try:
        data = yf.download(list(ticker_list), start=START_DATE, end=END_DATE, auto_adjust=True)["Close"]
        if data.empty:
            raise ValueError("No data returned from API.")
        data = data[list(ticker_list)]
        return data.dropna()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@st.cache_data(ttl=86400)
def get_ecb_deposit_rate():
    """Returns (rate, source) so the app always shows where the rate came from."""
    try:
        url = ("https://sdw-wsrest.ecb.europa.eu/service/data/FM/"
               "B.U2.EUR.4F.KR.DFR_LEV.LEV?lastNObservations=1&format=csvdata")
        df = pd.read_csv(url)
        if not df.empty and "OBS_VALUE" in df.columns:
            return float(df["OBS_VALUE"].iloc[-1]) / 100, "ECB (live)"
    except Exception:
        pass
    return FALLBACK_RF, "Fallback (ECB fetch failed)"

def exact_max_sharpe(mu, cov, rf):
    k = len(mu)
    def neg_sharpe(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        return -(ret - rf) / vol

    res = minimize(neg_sharpe,
                   x0=np.ones(k) / k, method="SLSQP",
                   bounds=[(0, 1)] * k,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return res.x

# --- PHASE 3: STEP 1, UI SETUP ---
st.title("Portfolio Optimiser")
st.write("This tool calculates the best asset allocation based on historical risk and return.")

st.sidebar.header("Optimiser Settings")

available_options = ["AAPL", "MSFT", "JNJ", "VTI", "BND", "TSLA", "GOOGL", "AMZN"]
selected_tickers = st.sidebar.multiselect(
    "Select assets to include:",
    options=available_options,
    default=["AAPL", "MSFT", "JNJ", "VTI", "BND"]
)

num_simulations = st.sidebar.slider(
    "Number of portfolios to simulate:", 
    min_value=100, 
    max_value=5000, 
    value=5000
)

run_button = st.sidebar.button("Run Optimisation")

# --- PHASE 3, TASK 2: MONTE CARLO SIMULATION & SOLVER ---
if run_button:
    if len(selected_tickers) < 2:
        st.error("Please select at least 2 assets.")
    else:
        st.write("🔄 Fetching latest data and running simulations...")
        data = get_real_data(selected_tickers)
        
        if data is None or data.empty:
            st.error("Could not find data for these tickers. Please try again.")
        else:
            daily_returns = data.pct_change().dropna()
            current_returns = daily_returns.mean() * 252
            cov_matrix = daily_returns.cov() * 252
            rf, rf_source = get_ecb_deposit_rate()

            # Seeded simulation
            rng = np.random.default_rng(SEED)
            k = len(selected_tickers)
            all_weights = rng.dirichlet(np.ones(k), size=num_simulations)
            
            ret_arr = all_weights @ current_returns.values
            vol_arr = np.sqrt(np.einsum("ij,jk,ik->i", all_weights, cov_matrix.values, all_weights))
            sharpe_arr = (ret_arr - rf) / vol_arr

            sim_data = {
                'Return': ret_arr,
                'Risk': vol_arr,
                'Sharpe': sharpe_arr
            }
            sim_df = pd.DataFrame(sim_data)
            
            st.success("Simulations Complete!") 

            # --- PHASE 3, TASK 3: IDENTIFYING THE BEST PORTFOLIO & SOLVER ---
            max_sharpe_idx = sim_df['Sharpe'].idxmax()
            best_ret = sim_df.loc[max_sharpe_idx, 'Return']
            best_vol = sim_df.loc[max_sharpe_idx, 'Risk']
            best_sharpe = sim_df.loc[max_sharpe_idx, 'Sharpe']
            best_weights = all_weights[max_sharpe_idx]

            # Mathematical solver
            w_exact = exact_max_sharpe(current_returns.values, cov_matrix.values, rf)
            e_ret = float(w_exact @ current_returns.values)
            e_vol = float(np.sqrt(w_exact @ cov_matrix.values @ w_exact))
            e_sharpe = (e_ret - rf) / e_vol

            st.subheader("🏆 Optimal Portfolio Found (Simulation)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Expected Return", f"{best_ret:.2%}")
            col2.metric("Annual Risk", f"{best_vol:.2%}")
            col3.metric("Sharpe Ratio", f"{best_sharpe:.2f}")
            col4.metric("Risk-free rate", f"{rf:.2%}", rf_source)

            st.subheader("🎯 Exact Max-Sharpe (Solver)")
            d1, d2, d3 = st.columns(3)
            d1.metric("Expected Return", f"{e_ret:.2%}")
            d2.metric("Annual Risk", f"{e_vol:.2%}")
            d3.metric("Sharpe Ratio", f"{e_sharpe:.2f}")

            st.write(f"Data window: {START_DATE} to {END_DATE} | Seed: {SEED}")
            st.write("### Recommended Allocations:")
            allocation_df = pd.DataFrame({
                'Asset': selected_tickers,
                'Simulation (%)': (best_weights * 100).round(1),
                'Solver (%)': (w_exact * 100).round(1)
            })
            st.table(allocation_df)

            # --- PHASE 3, TASK 4: VISUALISATION ---
            st.write("### Portfolio Risk vs. Return")
            
            fig = px.scatter(
                sim_df, 
                x='Risk', 
                y='Return', 
                color='Sharpe',
                labels={'Risk': 'Annualised Risk (Volatility)', 'Return': 'Annualised Return'},
                title=f"Monte Carlo Simulation: Finding the Efficient Frontier (Risk-free rate {rf:.2%}, {rf_source})",
                color_continuous_scale='Viridis'
            )
            
            fig.add_scatter(
                x=[best_vol], 
                y=[best_ret], 
                mode="markers",
                marker=dict(color='red', size=15, symbol='star'),
                name="Best Simulated"
            )
            fig.add_scatter(
                x=[e_vol], 
                y=[e_ret], 
                mode="markers",
                marker=dict(color='white', size=13, symbol='diamond', line=dict(color='black', width=1)),
                name="Exact (Solver)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
