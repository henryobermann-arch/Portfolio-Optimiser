import yfinance as yf 
import pandas as pd    
import numpy as np    
import streamlit as st 
import plotly.express as px

# --- STEP 1: INITIAL DATA SETUP ---
assets = [
    {"ticker": "AAPL", "expected_return": 0.10, "risk_score": 5},
    {"ticker": "MSFT", "expected_return": 0.12, "risk_score": 4},
    {"ticker": "JNJ", "expected_return": 0.05, "risk_score": 2},
    {"ticker": "VTI", "expected_return": 0.08, "risk_score": 3},
    {"ticker": "BND", "expected_return": 0.03, "risk_score": 1}
]

# --- STEP 2: PORTFOLIO WEIGHTS ---
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# --- STEP 3: CALCULATING PORTFOLIO RETURN ---
portfolio_return = sum(asset["expected_return"] * weights[i] for i, asset in enumerate(assets))

# --- STEP 4: CALCULATING PORTFOLIO RISK ---
portfolio_risk = sum(asset["risk_score"] * weights[i] for i, asset in enumerate(assets))

# --- HELPER FUNCTIONS ---
def get_real_data(ticker_list):
    try:
        data = yf.download(ticker_list, period="3y")["Close"]
        if data.empty:
            raise ValueError("No data returned from API.")
        data = data[ticker_list]
        return data.dropna()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@st.cache_data(ttl=86400)
def get_ecb_deposit_rate():
    try:
        ecb_url = "https://sdw-wsrest.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.DFR_LEV.LEV?lastNObservations=1&format=csvdata"
        df = pd.read_csv(ecb_url)
        if not df.empty and 'OBS_VALUE' in df.columns:
            return round(float(df['OBS_VALUE'].iloc[-1]), 4) / 100
    except Exception:
        pass
    return 0.025  # Fallback: 2.50% ECB deposit rate

# --- UI SETUP ---
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
    value=1000
)

run_button = st.sidebar.button("Run Optimisation")

# --- MONTE CARLO SIMULATION ---
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

            risk_free_rate = get_ecb_deposit_rate()

            all_weights = []
            ret_arr = []
            vol_arr = []
            sharpe_arr = []

            for i in range(num_simulations):
                weights = np.array(np.random.random(len(selected_tickers)))
                weights = weights / np.sum(weights)
                all_weights.append(weights)

                p_ret = np.sum(current_returns.values * weights)
                ret_arr.append(p_ret)

                p_vol = np.sqrt(weights @ cov_matrix.values @ weights)
                vol_arr.append(p_vol)

                # Sharpe ratio formula subtracting the ECB rate
                p_sharpe = (p_ret - risk_free_rate) / p_vol
                sharpe_arr.append(p_sharpe)

            sim_data = {
                'Return': ret_arr,
                'Risk': vol_arr,
                'Sharpe': sharpe_arr
            }
            sim_df = pd.DataFrame(sim_data)
            
            st.success("Simulations Complete!") 

            # --- IDENTIFYING BEST PORTFOLIO ---
            max_sharpe_idx = sim_df['Sharpe'].idxmax()
            best_ret = sim_df.loc[max_sharpe_idx, 'Return']
            best_vol = sim_df.loc[max_sharpe_idx, 'Risk']
            best_sharpe = sim_df.loc[max_sharpe_idx, 'Sharpe']
            best_weights = all_weights[max_sharpe_idx]

            st.subheader("🏆 Optimal Portfolio Found")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Expected Return", f"{best_ret:.2%}")
            col2.metric("Annual Risk", f"{best_vol:.2%}")
            col3.metric("Sharpe Ratio", f"{best_sharpe:.2f}")
            col4.metric("ECB Risk-Free Rate", f"{risk_free_rate:.2%}")

            st.write("### Recommended Allocations:")
            allocation_df = pd.DataFrame({
                'Asset': selected_tickers,
                'Weight (%)': (best_weights * 100).round(1)
            })
            st.table(allocation_df)

            # --- VISUALISATION ---
            st.write("### Portfolio Risk vs. Return")
            
            fig = px.scatter(
                sim_df, 
                x='Risk', 
                y='Return', 
                color='Sharpe',
                labels={'Risk': 'Annualised Risk (Volatility)', 'Return': 'Annualised Return'},
                title=f"Monte Carlo Simulation: Efficient Frontier (ECB Rate: {risk_free_rate:.2%})",
                color_continuous_scale='Viridis'
            )
            
            fig.add_scatter(
                x=[best_vol], 
                y=[best_ret], 
                marker=dict(color='red', size=15, symbol='star'),
                name="Optimal Portfolio"
            )
            
            st.plotly_chart(fig, use_container_width=True)
