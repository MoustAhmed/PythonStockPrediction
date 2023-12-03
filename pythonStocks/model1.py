import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Get Historical Data
ticker_symbol = "MSFT"
start_date = "2020-01-01"
end_date = "2023-11-30"
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Feature Engineering
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['SMA_50'], label='SMA 50')
plt.plot(data['SMA_200'], label='SMA 200')
plt.title(f'{ticker_symbol} Closing Prices and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Prepare the Data for Machine Learning
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

# Ensure that the training data includes the date of the last data point in the chart
split_percentage = 0.8
split_index = int(split_percentage * len(data))

X = data[['Close', 'SMA_50', 'SMA_200']]
y = data['Target']

X_train, X_test = X[:split_index + 1], X[split_index + 1:]
y_train, y_test = y[:split_index + 1], y[split_index + 1:]

# Simple Prediction Model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Predictions for Future Prices
last_data = X.iloc[-1].values.reshape(1, -1)
next_day_prediction = model.predict(last_data)
print(f'Predicted Next Day Closing Price: {next_day_prediction[0]}')
