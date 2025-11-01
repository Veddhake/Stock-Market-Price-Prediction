import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

# --------------------------------------
# 1. Load Data
# --------------------------------------
st.header("📈 Auto ARIMA Stock Forecast")

ticker = st.text_input("Enter Stock Ticker", "KO")
interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=2)
start_date = "2015-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
close_prices = data['Close'].dropna()
st.line_chart(close_prices)

# --------------------------------------
# 2. Train-Test Split
# --------------------------------------
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[:train_size], close_prices[train_size:]

# --------------------------------------
# 3. Auto ARIMA Model
# --------------------------------------
st.subheader("🔧 Running Auto ARIMA (may take a few seconds...)")
stepwise_model = auto_arima(
    train,
    start_p=1, start_q=1,
    max_p=5, max_q=5,
    seasonal=True, m=12,      # for monthly; change m to 4 for weekly etc.
    start_P=0, start_Q=0,
    max_P=2, max_Q=2,
    d=None, D=1,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True
)
st.code(stepwise_model.summary().as_text())

# --------------------------------------
# 4. Fit Final SARIMA Model
# --------------------------------------
model = SARIMAX(train, order=stepwise_model.order, seasonal_order=stepwise_model.seasonal_order)
model_fit = model.fit(disp=False)

# --------------------------------------
# 5. Predict & Evaluate
# --------------------------------------
n_test = len(test)
preds = model_fit.forecast(steps=n_test)
rmse = np.sqrt(mean_squared_error(test, preds))
mae = mean_absolute_error(test, preds)
r2 = r2_score(test, preds)

st.subheader("📊 Model Evaluation")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R² Score", f"{r2:.2f}")

# --------------------------------------
# 6. Plot
# --------------------------------------
fig1 = plt.figure(figsize=(12, 5))
plt.plot(test.index, test.values, label="Actual")
plt.plot(test.index, preds, label="Predicted", linestyle="--")
plt.title("Auto ARIMA Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig1)

# --------------------------------------
# 7. Forecast Future
# --------------------------------------
num_future_days = st.number_input("Forecast Future Periods", 1, 365, 30)
future_forecast = model_fit.forecast(steps=num_future_days)
last_date = close_prices.index[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(num_future_days)]

fig2 = plt.figure(figsize=(12, 5))
plt.plot(future_dates, future_forecast, label="Future Forecast", linestyle="--", marker="o")
plt.title(f"{ticker} - Future Forecast ({num_future_days} steps)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

# --------------------------------------
# 8. Download Forecast CSV
# --------------------------------------
future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_forecast})
csv = future_df.to_csv(index=False)
st.download_button("📥 Download Forecast CSV", csv, file_name="auto_arima_forecast.csv", mime="text/csv")

st.success("✅ Auto ARIMA Forecast Complete!")
