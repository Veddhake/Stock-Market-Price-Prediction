import os
import io
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -------------------------------
# Utility Functions
# -------------------------------
def fetch_stock_data(symbol, start, end):
    data = yf.download(symbol, start, end)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, Y = [], []
    for i in range(100, len(scaled_data)):
        X.append(scaled_data[i-100:i])
        Y.append(scaled_data[i, 0])
    return np.array(X), np.array(Y), scaler

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape), Dropout(0.2),
        LSTM(50, return_sequences=True), Dropout(0.2),
        LSTM(50), Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model

def plot_moving_averages(data):
    ma_50 = data['Close'].rolling(50).mean()
    ma_100 = data['Close'].rolling(100).mean()
    ma_200 = data['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Closing Price')
    plt.plot(ma_50, label='MA50')
    plt.plot(ma_100, label='MA100')
    plt.plot(ma_200, label='MA200')
    plt.legend(), plt.xlabel("Time"), plt.ylabel("Price")
    return fig

def plot_predictions(dates, actual, predicted, title):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual Price')
    plt.plot(dates, predicted, label='Predicted Price', linestyle='dashed')
    plt.legend(), plt.xlabel("Date"), plt.ylabel("Price")
    plt.title(title)
    plt.xticks(rotation=45)
    return fig

def get_image_download(fig, filename):
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png")
    return st.download_button(label=f"Download {filename}", data=img_buffer.getvalue(), file_name=filename, mime="image/png")

# -------------------------------
# Streamlit App
# -------------------------------
st.header("\U0001F4C8 Stock Market Predictor")

# Inputs
stock = st.text_input("Enter Stock Symbol", "GOOG")
num_future_days = st.number_input("Predict Future Days", 1, 365, 30)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 20)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

start_date = "2020-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
data = fetch_stock_data(stock, start_date, end_date)

st.subheader("\U0001F4CA Stock Data")
st.write(data)

# Split
train_data = pd.DataFrame(data['Close'][0: int(len(data) * 0.80)])
test_data = pd.DataFrame(data['Close'][int(len(data) * 0.80):])

# Preprocessing
X_train, Y_train, scaler = preprocess_data(train_data)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model_path = f"model_{stock}.keras"

if not os.path.exists(model_path):
    st.subheader("\U0001F527 Training LSTM Model...")
    model = build_lstm_model((X_train.shape[1], 1))
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(model_path)
else:
    model = load_model(model_path)
    st.success("\u2705 Loaded pre-trained model")

# Test Preparation
past_100 = train_data.tail(100)
final_data = pd.concat([past_100, test_data], ignore_index=True)
final_scaled = scaler.transform(final_data)

X_test, Y_test = [], []
for i in range(100, final_scaled.shape[0]):
    X_test.append(final_scaled[i-100:i])
    Y_test.append(final_scaled[i, 0])
X_test, Y_test = np.array(X_test), np.array(Y_test)

# Predict
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
Y_test_rescaled = scaler.inverse_transform(Y_test.reshape(-1, 1))

# Prepare and display test prediction comparison table
test_pred_df = pd.DataFrame({
    "Date": data.index[-len(Y_test):],
    "Actual Price": Y_test_rescaled.flatten(),
    "Predicted Price": predicted_prices.flatten()
})

# Show last date used for prediction
last_test_date = test_pred_df["Date"].iloc[-1]
st.subheader(f"🗓️ Last Date in Test Set: {last_test_date.strftime('%Y-%m-%d')}")

# Show test prediction table in frontend
st.subheader("📋 Test Set Predictions (Actual vs Predicted)")
st.dataframe(test_pred_df.tail(10))  # show last 10 rows for brevity

# Metrics
rmse = np.sqrt(mean_squared_error(Y_test_rescaled, predicted_prices))
mae = mean_absolute_error(Y_test_rescaled, predicted_prices)
r2 = r2_score(Y_test_rescaled, predicted_prices)

st.subheader("\U0001F4C8 Model Evaluation")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R² Score", f"{r2:.2f}")

# Future Prediction
future_input = final_scaled[-100:]
future_predictions = []
for _ in range(num_future_days):
    future_input_reshaped = np.reshape(future_input, (1, 100, 1))
    future_pred = model.predict(future_input_reshaped)
    future_predictions.append(future_pred[0, 0])
    future_input = np.append(future_input, future_pred)[-100:].reshape(-1, 1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, num_future_days + 1)]

# Plots
st.subheader("\U0001F4CA Moving Averages")
st.pyplot(plot_moving_averages(data))

st.subheader("\U0001F4CA Predicted vs Actual")
st.pyplot(plot_predictions(data.index[-len(Y_test):], Y_test_rescaled, predicted_prices, "Prediction on Test Set"))

st.subheader(f"\U0001F4CA Future {num_future_days}-Day Forecast")
fig_future = plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_predictions, label="Future Predictions", marker='o', linestyle='--')
plt.legend(), plt.xlabel("Date"), plt.ylabel("Price")
st.pyplot(fig_future)

# Downloads
pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions.flatten()})
csv_buffer = io.StringIO()
pred_df.to_csv(csv_buffer, index=False)
st.download_button("Download Predictions CSV", data=csv_buffer.getvalue(), file_name="predictions.csv", mime="text/csv")

col1, col2, col3 = st.columns(3)
with col1:
    get_image_download(plot_moving_averages(data), "moving_averages.png")
with col2:
    get_image_download(plot_predictions(data.index[-len(Y_test):], Y_test_rescaled, predicted_prices, "Prediction on Test Set"), "predicted_vs_actual.png")
with col3:
    get_image_download(fig_future, "future_predictions.png")

st.success("\u2705 Prediction Completed!")
# --- Prepare test set predictions for download ---
test_pred_df = pd.DataFrame({
    "Date": data.index[-len(Y_test):],
    "Actual Price": Y_test_rescaled.flatten(),
    "Predicted Price": predicted_prices.flatten()
})

test_csv_buffer = io.StringIO()
test_pred_df.to_csv(test_csv_buffer, index=False)
st.download_button("Download Test Set Predictions CSV", data=test_csv_buffer.getvalue(),
                   file_name="test_predictions.csv", mime="text/csv")
