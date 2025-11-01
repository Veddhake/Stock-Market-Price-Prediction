import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io

# ---------------------------------------------
# STREAMLIT HEADER
# ---------------------------------------------
st.header("📈 Stock Market Predictor (LSTM Based)")

# User inputs
stock = st.text_input("Enter Stock Symbol (e.g., GOOG, AAPL, TCS.NS)", "GOOG")
num_future_days = st.number_input("Predict Future Days", min_value=1, max_value=365, value=30, step=1)
start, end = "2020-01-01", datetime.today().strftime("%Y-%m-%d")

# ---------------------------------------------
# FETCH STOCK DATA
# ---------------------------------------------
try:
    data = yf.download(stock, start, end)
    if data.empty:
        st.error("❌ No data found. Please check the stock symbol or your internet connection.")
        st.stop()
    st.subheader("📊 Stock Data")
    st.write(data.tail())
except Exception as e:
    st.error(f"❌ Error fetching stock data: {e}")
    st.stop()

# ---------------------------------------------
# MOVING AVERAGES
# ---------------------------------------------
ma_50 = data["Close"].rolling(50).mean()
ma_100 = data["Close"].rolling(100).mean()
ma_200 = data["Close"].rolling(200).mean()

# ---------------------------------------------
# TRAIN-TEST SPLIT
# ---------------------------------------------
data = data[["Close"]].dropna()
train_size = int(len(data) * 0.8)
data_train = data.iloc[:train_size]
data_test = data.iloc[train_size:]

if len(data_train) == 0 or len(data_test) == 0:
    st.error("❌ Training or test data is empty. The dataset is too small or invalid.")
    st.stop()

# ---------------------------------------------
# DATA SCALING
# ---------------------------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# ---------------------------------------------
# SEQUENCE CREATION FOR LSTM
# ---------------------------------------------
X_train, Y_train = [], []
for i in range(100, len(data_train_scaled)):
    X_train.append(data_train_scaled[i - 100:i])
    Y_train.append(data_train_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# ---------------------------------------------
# TRAIN LSTM MODEL
# ---------------------------------------------
st.subheader("🛠 Training LSTM Model... (may take 1-2 minutes)")

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)

# Save & reload to verify
model_path = "predict.keras"
model.save(model_path)

try:
    model = load_model(model_path)
    st.success("✅ Model trained and loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ---------------------------------------------
# PREPARE TEST DATA
# ---------------------------------------------
past_100_days = data_train.tail(100)
data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test_full)

X_test, Y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    X_test.append(data_test_scaled[i - 100:i])
    Y_test.append(data_test_scaled[i, 0])

X_test, Y_test = np.array(X_test), np.array(Y_test)

# ---------------------------------------------
# PREDICT TEST DATA
# ---------------------------------------------
predicted_prices = model.predict(X_test)
scale_factor = 1 / scaler.scale_[0]
predicted_prices = predicted_prices * scale_factor
Y_test = Y_test * scale_factor

# ---------------------------------------------
# FUTURE PREDICTION
# ---------------------------------------------
future_input = data_test_scaled[-100:]
future_predictions = []

for _ in range(num_future_days):
    future_input_reshaped = np.reshape(future_input, (1, 100, 1))
    future_pred = model.predict(future_input_reshaped)
    future_predictions.append(future_pred[0, 0])
    future_input = np.append(future_input, future_pred)[-100:].reshape(-1, 1)

future_predictions = np.array(future_predictions) * scale_factor
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, num_future_days + 1)]

# ---------------------------------------------
# PLOTTING SECTION
# ---------------------------------------------
st.subheader("📊 Moving Averages vs. Original Price")
fig1 = plt.figure(figsize=(12, 6))
plt.plot(data["Close"], "g", label="Closing Price")
plt.plot(ma_50, "r", label="MA50")
plt.plot(ma_100, "b", label="MA100")
plt.plot(ma_200, "purple", label="MA200")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
st.pyplot(fig1)

# Predicted vs Actual Prices
st.subheader("📊 Predicted vs. Actual Prices")
fig2 = plt.figure(figsize=(12, 6))
test_dates = data.index[-len(Y_test):]
plt.plot(test_dates, Y_test, "g", label="Original Price")
plt.plot(test_dates, predicted_prices, "r--", label="Predicted Price")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)
st.pyplot(fig2)

# Future predictions
st.subheader(f"📊 Future {num_future_days}-Day Prediction")
fig3 = plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(predicted_prices):], predicted_prices, "orange", linestyle="dashed", label="Predicted Price (Test)")
plt.plot(future_dates, future_predictions, "red", linestyle="dashed", marker="o", label=f"Future {num_future_days}-Day Forecast")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Price")
st.pyplot(fig3)

# ---------------------------------------------
# DOWNLOAD SECTION
# ---------------------------------------------
st.subheader("📥 Download Prediction Data")

predictions_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": future_predictions
})

csv_buffer = io.StringIO()
predictions_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="Download Predictions CSV",
    data=csv_buffer.getvalue(),
    file_name="predictions.csv",
    mime="text/csv"
)

# ---------------------------------------------
# DOWNLOAD GRAPHS
# ---------------------------------------------
st.subheader("📥 Download Graphs")

def get_image_download(fig, filename):
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png")
    return st.download_button(
        label=f"Download {filename}",
        data=img_buffer.getvalue(),
        file_name=filename,
        mime="image/png"
    )

col1, col2, col3 = st.columns(3)
with col1:
    get_image_download(fig1, "moving_averages.png")
with col2:
    get_image_download(fig2, "predicted_vs_actual.png")
with col3:
    get_image_download(fig3, "future_predictions.png")

st.success("✅ Prediction Completed Successfully!")
