import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import datetime as dt
import requests
import yfinance as yf
from datetime import datetime, timedelta
from PIL import Image

st.set_page_config(page_title="Bitcoin Price Prediction", layout="wide")

st.markdown("""
    <style>
        .big-font { font-size:20px !important; font-weight:bold; }
        .red-text { color: #FF4B4B; }
        .green-text { color: #57D131; }
        .stApp { background-color: #f7f9fa; }
    </style>
""", unsafe_allow_html=True)

if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

SEQ_LENGTH = 10

st.sidebar.header("⚙️ Settings")
page = st.sidebar.selectbox("SELECT PAGE", ["🏠 Home", "📊 EDA", "💰 Live Bitcoin Price", "📈 Analysis", "🔮 Prediction", "ℹ️ About"])
prediction_days = st.sidebar.slider("📅 Days to Predict", min_value=1, max_value=30, value=10)

@st.cache_data
def fetch_data():
    try:
        btc = yf.Ticker("BTC-USD")
        end_date = datetime.now()
        data = btc.history(start="2013-04-28", end=end_date).reset_index()
        data = data.rename(columns={
            'Date': 'date', 'Open': 'open', 'High': 'high',
            'Low': 'low', 'Close': 'close', 'Volume': 'volume'
        })
        data['adj_close'] = data['close']
        data['date'] = pd.to_datetime(data['date'])
        data['Returns'] = data['close'].pct_change()
        data['MA7'] = data['close'].rolling(window=7).mean()
        data['MA14'] = data['close'].rolling(window=14).mean()
        data['MA30'] = data['close'].rolling(window=30).mean()
        data['Volatility'] = data['Returns'].rolling(window=30).std()
        return data[['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'Returns', 'MA7', 'MA14', 'MA30', 'Volatility']]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

if page == "🏠 Home":
    st.title("🏠 Welcome to the Bitcoin Price Prediction App ")
    logo = Image.open("bitcoin_chart.webp")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo, width=700)
    st.write("""
    - This app uses machine learning (XGBoost) to predict Bitcoin prices based on historical data.
    - Data is fetched from Yahoo Finance (2014 to present).
    """)

elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    data = fetch_data()
    if data is not None:
        st.subheader("🧾 Dataset Preview")
        st.dataframe(data.head())
        st.dataframe(data.tail())
        st.subheader("🧹 Null Values Check")
        st.write(data.isnull().sum())
        st.subheader("📊 Dataset Statistics")
        st.write(data.describe())
        st.subheader("📈 Bitcoin Closing Price Over Time")
        fig = px.line(data, x='date', y='close', title="Bitcoin Closing Price Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Failed to fetch data.")

elif page == "💰 Live Bitcoin Price":
    st.title("💰 Live Bitcoin Price")
    st.subheader("🔄 Bitcoin Live Price Checker")

    def get_bitcoin_price():
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd,inr"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return data['bitcoin']['usd'], data['bitcoin']['inr']
        except Exception as e:
            st.error(f"Error fetching price: {e}")
            return None, None

    if st.button("Get Current Bitcoin Price 💰"):
        usd_price, inr_price = get_bitcoin_price()
        if usd_price is not None:
            st.success(f"💵 Price in USD: ${usd_price:,.2f}")
            st.success(f"💵 Price in INR: ₹{inr_price:,.2f}")
        else:
            st.error("Failed to fetch Bitcoin price.")

    st.subheader("🧾Bitcoin Daliy Data")
    if st.button("  💹 Prediction Data  "):
        processed_data = fetch_data()
        if processed_data is not None:
            latest_data = processed_data.iloc[-1]
            volume = st.number_input("Volume", value=float(latest_data['volume']), min_value=0.0)
            returns = st.number_input("Daily Returns", value=float(latest_data['Returns']) if not pd.isna(latest_data['Returns']) else 0.0)
            ma7 = st.number_input("7-day MA", value=float(latest_data['MA7']) if not pd.isna(latest_data['MA7']) else 0.0, min_value=0.0)
            ma14 = st.number_input("14-day MA", value=float(latest_data['MA14']) if not pd.isna(latest_data['MA14']) else 0.0, min_value=0.0)
            ma30 = st.number_input("30-day MA", value=float(latest_data['MA30']) if not pd.isna(latest_data['MA30']) else 0.0, min_value=0.0)
            volatility = st.number_input("Volatility", value=float(latest_data['Volatility']) if not pd.isna(latest_data['Volatility']) else 0.0)

elif page == "📈 Analysis":
    st.title("📈 Model Analysis")
    data = fetch_data()
    if data is not None:
        closedf = data[['close']].dropna()
        scaler = MinMaxScaler(feature_range=(0, 1))
        closedf_scaled = scaler.fit_transform(closedf)
        st.session_state.scaler = scaler

        train_size = int(len(closedf_scaled) * 0.8)
        train_data, test_data = closedf_scaled[:train_size], closedf_scaled[train_size:]

        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
        X_test, y_test = create_sequences(test_data, SEQ_LENGTH)

        model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=5)
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        st.session_state.model = model

        test_preds = model.predict(X_test.reshape(X_test.shape[0], -1))
        test_preds_inv = scaler.inverse_transform(test_preds.reshape(-1, 1))
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        mse = mean_squared_error(y_test_inv, test_preds_inv)
        rmse = np.sqrt(mse)

        st.subheader("Model Performance")
        st.markdown(f"<p class='big-font green-text'>✅ Model Trained Successfully!</p>", unsafe_allow_html=True)
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        st.subheader("Test Data Predictions")
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(y=y_test_inv.flatten(), mode='lines', name='Actual'))
        fig_test.add_trace(go.Scatter(y=test_preds_inv.flatten(), mode='lines', name='Predicted'))
        fig_test.update_layout(title="Actual vs Predicted (Test Data)", xaxis_title="Time", yaxis_title="Close Price")
        st.plotly_chart(fig_test, use_container_width=True)
    else:
        st.warning("⚠️ Failed to fetch data.")

elif page == "🔮 Prediction":
    st.title(f"🔮 Prediction for Next {prediction_days} Days")
    data = fetch_data()
    if data is not None and st.session_state.model and st.session_state.scaler:
        closedf = data[['close']].dropna()
        scaler = st.session_state.scaler
        closedf_scaled = scaler.transform(closedf)
        last_sequence = closedf_scaled[-SEQ_LENGTH:]

        predictions = []
        current_seq = last_sequence.copy()

        for _ in range(prediction_days):
            next_pred = st.session_state.model.predict(current_seq.reshape(1, -1))
            predictions.append(next_pred[0])
            current_seq = np.append(current_seq[1:], next_pred[0]).reshape(-1, 1)

        predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = [data['date'].iloc[-1] + dt.timedelta(days=i) for i in range(1, prediction_days + 1)]

        fig_pred = px.line(x=future_dates, y=predictions_inv.flatten(), title=f"Predicted Bitcoin Price for Next {prediction_days} Days")
        fig_pred.update_layout(xaxis_title="Date", yaxis_title="Predicted Close Price")
        st.plotly_chart(fig_pred, use_container_width=True)

        @st.cache_data
        def load_data():
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*2)
                btc = yf.download('BTC-USD', start=start_date, end=end_date)
                if btc.empty:
                    raise ValueError("No valid data fetched.")
                
                if isinstance(btc.columns, pd.MultiIndex):
                    btc = btc.xs('BTC-USD', level=1, axis=1, drop_level=True)
                elif 'Close' not in btc.columns:
                    raise ValueError(f"Missing 'Close' column. Available columns: {list(btc.columns)}")
                return btc
            except Exception as e:
                st.error(f"Error fetching data: {e}")
                return None

        def prepare_features(df):
            if 'Close' not in df.columns:
                raise ValueError(f"DataFrame missing 'Close' column. Available columns: {list(df.columns)}")
            result_df = df.copy()
            result_df['Target'] = result_df['Close'].shift(-1)
            if 'Target' not in result_df.columns:
                raise ValueError("Failed to create 'Target' column.")
            return result_df

        def train_model(X, y):
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    missing=np.nan
                )
                model.fit(X_train, y_train)
                return model
            except Exception as e:
                st.error(f"Error training model: {e}")
                return None

        def main():
            with st.spinner("Loading data..."):
                btc_data = load_data()
                if btc_data is None:
                    return
                
                try:
                    processed_data = prepare_features(btc_data)
                except ValueError as e:
                    st.error(f"Error preparing features: {e}")
                    return
                
                if processed_data is None or processed_data.empty:
                    st.error("No valid data.")
                    return
                
                if 'Target' not in processed_data.columns:
                    st.error(f"Target column missing. Available columns: {list(processed_data.columns)}")
                    return
            
                try:
                    processed_data = processed_data.dropna(subset=['Target'])
                except KeyError as e:
                    st.error(f"Error filtering data: {e}. Available columns: {list(processed_data.columns)}")
                    return
                
                if processed_data.empty:
                    st.error("No valid data after filtering.")
                    return
            
            features = ['Close']
            X = processed_data[features]
            y = processed_data['Target']
            
            with st.spinner("Training model..."):
                model = train_model(X, y)
                if model is None:
                    return
            
            latest_data = processed_data.iloc[-1]
            close_price = st.number_input("Current Price", value=float(latest_data['Close']), min_value=0.0)

            if st.button("Predict Next Day's Price"):
                try:
                    input_data = np.array([[close_price]])
                    prediction = model.predict(input_data)[0]
                    
                    st.success(f"Predicted Bitcoin Price for next day: ${prediction:,.2f}")
                    
                    price_diff = prediction - close_price
                    diff_percent = (price_diff / close_price) * 100
                    if price_diff > 0:
                        st.write(f"📈 Expected increase: ${price_diff:,.2f} ({diff_percent:.2f}%)")
                    else:
                        st.write(f"📉 Expected decrease: ${abs(price_diff):,.2f} ({abs(diff_percent):.2f}%)")
                except Exception as e:
                    st.error(f"Error predicting: {e}")

        if __name__ == "__main__":
            main()
            
    
                
    else:
        st.warning("⚠️ Train the model first from the Analysis page.")

elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    ### Overview
    This application demonstrates Bitcoin price prediction using machine learning (**XGBoost**).
    
    ### Features
    - Fetches Bitcoin historical data from Yahoo Finance.
    - Performs exploratory data analysis and visualization.
    - Predicts future prices using a trained XGBoost model.

    ### Technologies Used
    - Streamlit, Pandas, NumPy, Plotly
    - Scikit-learn, XGBoost, yFinance, CoinGecko API
    """)
