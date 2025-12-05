import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import plotly.graph_objects as go # You might need to: python -m pip install plotly

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="AI Market Predictor", layout="wide")
st.title("🤖 AI Stock Swing Trader")
st.markdown("Enter a ticker symbol to train a Random Forest model on its history and predict the next 5 days.")

# --- 2. SIDEBAR INPUTS ---
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()
horizon = 5 # Predicting 5 days out

# --- 3. THE ENGINE (FUNCTIONS) ---
def get_data(ticker):
    """Fetches data and calculates MACD, RSI, MA50"""
    try:
        # Get data including today
        df = yf.download(ticker, period="5y", progress=False)
        
        # Handle MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        else:
            df = df[['Close']]
            
        df = df.rename(columns={'Close': 'Close', ticker: 'Close'})
        
        # --- FEATURE ENGINEERING ---
        # 1. MACD (Momentum)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = macd - signal
        
        # 2. RSI (Momentum)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. Moving Average Distance (Value)
        df['Dist_MA50'] = (df['Close'] / df['Close'].rolling(50).mean()) - 1
        
        # 4. Target (Did it go up next week?)
        # We target a simple positive return for the general tool
        df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error finding ticker {ticker}. Try specific format like BTC-USD.")
        return None

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "52 Wk High": info.get("fiftyTwoWeekHigh", "N/A"),
            "Sector": info.get("sector", "N/A")
        }
    except:
        return None

def train_and_predict(data):
    # Split Data (Train on old, Test on recent)
    # We use everything EXCEPT the last row for training
    # The last row is "Today", which we need to predict
    
    predictors = ['MACD_Hist', 'RSI', 'Dist_MA50']
    X = data[predictors]
    y = data['Target']
    
    # Train on all available data where we have a known target
    # (We can't train on the last 5 days because we don't know the future yet)
    train_max_idx = len(data) - horizon
    
    X_train = X.iloc[:train_max_idx]
    y_train = y.iloc[:train_max_idx]
    
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
    model.fit(X_train, y_train)
    
    # Predict for "Today" (The most recent data point)
    current_features = X.iloc[[-1]]
    prediction = model.predict(current_features)[0]
    probability = model.predict_proba(current_features)[0][1]
    
    return prediction, probability, model

# --- 4. THE MAIN EXECUTION ---
if st.button("Generate Prediction"):
    with st.spinner(f"Analyzing {ticker}..."):
        data = get_data(ticker)
        
        if data is not None:
            # ... inside the main loop ...
            fund_data = get_fundamentals(ticker)
            if fund_data:
                st.sidebar.markdown("### 📊 Fundamental Data")
                st.sidebar.write(f"**Sector:** {fund_data['Sector']}")
                # Format large numbers to be readable (e.g., 1T)
                st.sidebar.write(f"**Market Cap:** {fund_data['Market Cap']}") 
                st.sidebar.write(f"**P/E Ratio:** {fund_data['P/E Ratio']}")
            # Show the Raw Data Chart
            st.subheader(f"{ticker} Price Chart (5 Years)")
            # Create a CandleStick Chart
            fig = go.Figure(data=[go.Candlestick(x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'])])
            
            fig.update_layout(title=f"{ticker} Interactive Chart", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Run the AI
            pred, prob, model = train_and_predict(data)
            
            # Display Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction (Next 5 Days)")
                if pred == 1:
                    st.success(f"**BUY / BULLISH**")
                else:
                    st.error(f"**SELL / BEARISH**")
                    
            with col2:
                st.subheader("Model Confidence")
                st.metric(label="Probability of Rise", value=f"{prob:.1%}")
            
            # Feature Importance
            st.subheader("Why did the AI decide this?")
            predictors = ['MACD_Hist', 'RSI', 'Dist_MA50']
            importances = pd.DataFrame({
                'Feature': predictors,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            st.bar_chart(importances.set_index('Feature'))
            
            # Disclaimer
            st.warning("⚠️ Disclaimer: This is an educational project. Not financial advice.")

else:
    st.info("Enter a ticker on the left and click 'Generate Prediction'")


