import os
import io
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_features(data):
    df = data.copy()
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['DayOfWeek'] = df.index.dayofweek
    df['Lag1'] = df['Close'].shift(1)
    df['Lag2'] = df['Close'].shift(2)
    df['Lag3'] = df['Close'].shift(3)
    df['Lag4'] = df['Close'].shift(4)
    df.dropna(inplace=True)
    return df



st.header("Stock Price Predictor (Random Forest)")

stock = st.text_input("Enter Stock Symbol", "GOOG")
num_future_days = st.number_input("Predict Future Days", 1, 365, 30)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 30, 10)

start_date = "2024-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")
data = yf.download(stock, start=start_date, end=end_date)

st.subheader("Stock Data")
st.write(data.tail())

df = create_features(data)
features = ['Day', 'Month', 'Year', 'DayOfWeek', 'Lag1', 'Lag2', 'Lag3', 'Lag4']
X = df[features]
y = df['Close']

split_index = int(len(df) * 0.8)
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]
dates_test = df.index[split_index:]

model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Evaluation")
col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R² Score", f"{r2:.2f}")

fig1 = plt.figure(figsize=(12, 6))
plt.plot(dates_test, y_test, label="Actual")
plt.plot(dates_test, y_pred, label="Predicted", linestyle='--')
plt.legend(), plt.xlabel("Date"), plt.ylabel("Price")
plt.title("Prediction on Test Set")
plt.xticks(rotation=45)
st.pyplot(fig1)

# Future Forecasting (using only lag-based features)

st.subheader(f"Future {num_future_days}-Day Forecast")

future_preds = []
future_dates = []
last_known = df.copy()

for i in range(num_future_days):
    next_date = data.index[-1] + timedelta(days=i+1)
    last_row = last_known.iloc[-1].copy()

    row = {
        'Day': next_date.day,
        'Month': next_date.month,
        'Year': next_date.year,
        'DayOfWeek': next_date.weekday(),
        'Lag1': last_row['Close'],
        'Lag2': last_row['Lag1'],
        'Lag3': last_row['Lag2'],
        'Lag4': last_row['Lag3'],
    }

    input_df = pd.DataFrame([row])[features]
    pred = model.predict(input_df)[0]

    future_preds.append(pred)
    future_dates.append(next_date)

    new_row = row.copy()
    new_row['Close'] = pred
    last_known.loc[next_date] = new_row

fig2 = plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_preds, label="Future Predictions", marker='o', linestyle='--')
plt.legend(), plt.xlabel("Date"), plt.ylabel("Price")
st.pyplot(fig2)

pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_preds})
csv_buffer = io.StringIO()
pred_df.to_csv(csv_buffer, index=False)
st.download_button("Download Predictions CSV", data=csv_buffer.getvalue(), file_name="rf_predictions.csv", mime="text/csv")

st.success("Prediction Completed!")


# produces better results when strated from 2024-01-01
# flat lining for long term data