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
     # Order matches: AAPL, MSFT, JNJ, VTI, BND

weights = [0.2,0.2,0.2,0.2,0.2]

# --- STEP 3: CALCULATING PORTFOLIO RETURN ---

portfolio_return = 0
for i, asset in enumerate(assets):
    contribution = asset["expected_return"] * weights[i]
    portfolio_return += contribution
   

# --- Step 4: Calculating portfolio risk --

portfolio_risk = 0

for i, asset in enumerate(assets):
    portfolio_risk += asset["risk_score"] * weights[i]


# --- Phase 2: Real data Integration ---

ticker_list = [asset["ticker"] for asset in assets]
print(f"Fetching 3 years of data for: {ticker_list}")

raw_data = yf.download(ticker_list, period="3y")

if raw_data.empty:
    print("Error: No data found.")
    exit()

print("Successfully fetched real price data!")

# --- Phase 2, task 2: Real Returns Calculation ---

close_prices = raw_data["Close"]
daily_returns = close_prices.pct_change()
annual_returns = daily_returns.mean() * 252

print("Annualized expected returns based on 3y history:")
print(annual_returns)

# --- Phase 2, task 3: Real volatility calculation ---

daily_returns = close_prices.pct_change()
annual_volatility = daily_returns.std() * np.sqrt(252)

print("Annualized volatility (real risk):")
print(annual_volatility)

# --- Phase 2: Step 4, Overwriting with real data ---

for asset in assets:
    ticker_name = asset["ticker"] 
    
asset["expected_return"] = annual_returns[ticker_name] 
asset["risk_score"] = annual_volatility[ticker_name] 

print("Successfully integrated real market data into the asset list.")
print(f"New data for {assets[0]['ticker']}: Return={assets[0]['expected_return']:.2%}, Risk={assets[0]['risk_score']:.2%}")

# --- Phase 2: Step 5, Error handling --- 

def get_real_data(ticker_list):
    try:
        
        data = yf.download(ticker_list, period="3y")["Close"]
        if data.empty:
            raise ValueError("No data returned from API.")
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# --- Phase 3: step 1, UI setup ---


st.title("Portfolio Optimizer")
st.write("This tool calculates the best asset allocation based on historical risk and return.")


st.sidebar.header("Optimizer Settings")


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

run_button = st.sidebar.button("Run Optimization")

# --- PHASE 3, TASK 2: MONTE CARLO SIMULATION ---

if run_button:
    st.write("🔄 Fetching latest data and running simulations...")
    data = yf.download(selected_tickers, period="3y")["Close"]
    
    if data.empty:
        st.error("Could not find data for these tickers. Please try again.")
    else:
        daily_returns = data.pct_change()
        current_returns = daily_returns.mean() * 252
        current_vol = daily_returns.std() * np.sqrt(252)

        all_weights = []
        ret_arr = []
        vol_arr = []
        sharpe_arr = []

        
        for i in range(num_simulations):
            weights = np.array(np.random.random(len(selected_tickers)))
            weights = weights / np.sum(weights)
            all_weights.append(weights)

            p_ret = np.sum(current_returns * weights)
            ret_arr.append(p_ret)

            p_vol = np.sum(current_vol * weights)
            vol_arr.append(p_vol)

            sharpe_arr.append(p_ret / p_vol)

        
        sim_data = {
            'Return': ret_arr,
            'Risk': vol_arr,
            'Sharpe': sharpe_arr
        }
        sim_df = pd.DataFrame(sim_data)
        
        st.success("Simulations Complete!") 

 # --- PHASE 3, TASK 3: IDENTIFYING THE BEST PORTFOLIO ---

    max_sharpe_idx = sim_df['Sharpe'].idxmax()
    best_ret = sim_df.loc[max_sharpe_idx, 'Return']
    best_vol = sim_df.loc[max_sharpe_idx, 'Risk']
    best_weights = all_weights[max_sharpe_idx]


    st.subheader("🏆 Optimal Portfolio Found")
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{best_ret:.2%}")
    col2.metric("Annual Risk", f"{best_vol:.2%}")
    col3.metric("Sharpe Ratio", f"{sim_df.loc[max_sharpe_idx, 'Sharpe']:.2f}")

    st.write("### Recommended Allocations:")
    allocation_df = pd.DataFrame({
        'Asset': selected_tickers,
        'Weight': best_weights
    })
    st.table(allocation_df)

# --- PHASE 3, TASK 4: VISUALIZATION ---
    st.write("### Portfolio Risk vs. Return")
    
    fig = px.scatter(
    sim_df, 
    x='Risk', 
    y='Return', 
    color='Sharpe',
    labels={'Risk': 'Annualized Risk (Volatility)', 'Return': 'Annualized Return'},
    title="Monte Carlo Simulation: Finding the Efficient Frontier",
    color_continuous_scale='Viridis'
    )
    
    fig.add_scatter(
    x=[best_vol], 
    y=[best_ret], 
    marker=dict(color='red', size=15, symbol='star'),
    name="Optimal Portfolio"
    )
    
    st.plotly_chart(fig, use_container_width=True)

