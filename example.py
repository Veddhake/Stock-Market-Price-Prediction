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

# Streamlit Header
st.header("📈 Stock Market Predictor")

# User Input for Stock & Prediction Days
stock = st.text_input("Enter Stock Symbol (e.g., GOOG, AAPL, MSFT)", "GOOG")
num_future_days = st.number_input("Predict Future Days", min_value=1, max_value=365, value=30, step=1)
start, end = "2020-01-01", datetime.today().strftime("%Y-%m-%d")

# Fetch Stock Data
try:
    data = yf.download(stock, start, end)
    st.subheader("📊 Stock Data")
    st.write(data)
except Exception as e:
    st.error(f"❌ Error fetching stock data: {e}")
    st.stop()

# Moving Averages
ma_50, ma_100, ma_200 = data["Close"].rolling(50).mean(), data["Close"].rolling(100).mean(), data["Close"].rolling(200).mean()

# Train-Test Split
data_train = pd.DataFrame(data["Close"][0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data["Close"][int(len(data) * 0.80):])

#  Preprocessing for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# Create sequences for LSTM
X_train, Y_train = [], []
for i in range(100, len(data_train_scaled)):
    X_train.append(data_train_scaled[i-100:i])
    Y_train.append(data_train_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Train LSTM Model
st.subheader("🛠 Training LSTM Model...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)), Dropout(0.2),
    LSTM(50, return_sequences=True), Dropout(0.2),
    LSTM(50), Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)

# Save & Load Model
model_path = "predict.keras"
model.save(model_path)
try:
    model = load_model(model_path)
    st.success("✅ Model trained & loaded!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Prepare Test Data
past_100_days = data_train.tail(100)
data_test_full = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test_full)

# sequences for LSTM test
X_test, Y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    X_test.append(data_test_scaled[i-100:i])
    Y_test.append(data_test_scaled[i, 0])
X_test, Y_test = np.array(X_test), np.array(Y_test)

# Make Predictions
predicted_prices = model.predict(X_test)
scale_factor = 1 / scaler.scale_
predicted_prices, Y_test = predicted_prices * scale_factor, Y_test * scale_factor

# Future Predictions
future_input = data_test_scaled[-100:]
future_predictions = []

for _ in range(num_future_days):
    future_input_reshaped = np.reshape(future_input, (1, 100, 1))
    future_pred = model.predict(future_input_reshaped)
    future_predictions.append(future_pred[0, 0])
    future_input = np.append(future_input, future_pred)[-100:].reshape(-1, 1)

future_predictions = np.array(future_predictions) * scale_factor
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, num_future_days + 1)]

# Moving Averages vs. Original Prices
st.subheader("📊 Moving Averages vs. Original Price")
fig1 = plt.figure(figsize=(12, 6))
plt.plot(data["Close"], "g", label="Closing Price")
plt.plot(ma_50, "r", label="MA50")
plt.plot(ma_100, "b", label="MA100")
plt.plot(ma_200, "purple", label="MA200")
plt.legend(), plt.xlabel("Time"), plt.ylabel("Price")
st.pyplot(fig1)

# Plot: Predicted vs. Original Prices
st.subheader("📊 Predicted vs. Original Prices")

fig2 = plt.figure(figsize=(12, 6))


test_dates = data.index[-len(Y_test):]  # Get corresponding dates for test data

plt.plot(test_dates, Y_test, "g", label="Original Price")
plt.plot(test_dates, predicted_prices, "r", linestyle="dashed", label="Predicted Price")

plt.legend()
plt.xlabel("Year")
plt.ylabel("Price")
plt.xticks(rotation=45)  # Rotate dates for better readability
st.pyplot(fig2)

# Plot: Future Predictions
st.subheader(f"📊 Future {num_future_days}-Day Prediction")
fig3 = plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(predicted_prices):], predicted_prices, "orange", linestyle="dashed", label="Predicted Price (Test Set)")
plt.plot(future_dates, future_predictions, "red", linestyle="dashed", marker="o", label=f"Future {num_future_days}-Day Prediction")
plt.legend(), plt.xlabel("Time"), plt.ylabel("Price")
st.pyplot(fig3)

st.success("✅ Prediction Completed!")

st.subheader("📥 Download Prediction Data")
predictions_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions})
csv_buffer = io.StringIO()
predictions_df.to_csv(csv_buffer, index=False)
st.download_button(label="Download Predictions CSV", data=csv_buffer.getvalue(), file_name="predictions.csv", mime="text/csv")


# Download Graphs
st.subheader("📥 Download Graphs")

# Convert figures to byte stream
def get_image_download(fig, filename):
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png")
    return st.download_button(label=f"Download {filename}", data=img_buffer.getvalue(), file_name=filename, mime="image/png")

col1, col2, col3 = st.columns(3)
with col1:
    get_image_download(fig1, "moving_averages.png")
with col2:
    get_image_download(fig2, "predicted_vs_original.png")
with col3:
    get_image_download(fig3, "future_predictions.png")

st.success("✅ Prediction Completed!")