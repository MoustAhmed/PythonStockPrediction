import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from ta.momentum import RSIIndicator  # Make sure to install the 'ta' library using 'pip install ta'
import numpy as np
from sklearn.metrics import mean_absolute_error

# Get user input for the stock symbol
ticker_symbol = input("Enter the stock symbol (e.g., MSFT): ").upper()

start_date = "2020-01-01"
end_date = "2023-11-16"
data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Feature Engineering
data['SMA_50'] = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Calculate RSI
rsi_window = 14
data['RSI'] = RSIIndicator(data['Close'], window=rsi_window).rsi()

plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['SMA_50'], label='SMA 50')
plt.plot(data['SMA_200'], label='SMA 200')
plt.plot(data['RSI'], label='RSI', linestyle='--')
plt.title(f'{ticker_symbol} Closing Prices, Moving Averages, and RSI')
plt.xlabel('Date')
plt.ylabel('Price/RSI')
plt.legend()
plt.show()

# Prepare the Data for Machine Learning
data['Target'] = data['Close'].shift(-1)
data = data.dropna()

# Ensure that the training data includes the date of the last data point in the chart
split_percentage = 0.9
split_index = int(split_percentage * len(data))

X = data[['Close', 'SMA_50', 'SMA_200', 'RSI']]  # Include RSI in the features
y = data['Target']

X_train, X_test = X[:split_index + 1], X[split_index + 1:]
y_train, y_test = y[:split_index + 1], y[split_index + 1:]

# Simple Prediction Model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)

# Predictions for Future Prices
last_data = X.iloc[-1].values.reshape(1, -1)
next_day_prediction = model.predict(last_data)
# Calculate the predicted prices for the next day
next_day_predictions = model.predict(X)

# Shift the predictions and actual prices to align them on the 'next day'
shifted_predictions = np.roll(next_day_predictions, -1)
shifted_actual_prices = np.roll(y.values, -1)

# Calculate the average accuracy of the predictions
accuracy = np.mean(np.abs((y_test - predictions) / y_test)) * 100
avg_accuracy = 100 - accuracy

print(f'Predicted Next Day Closing Price for {ticker_symbol}: {next_day_prediction[0]}')
print(f'Average Accuracy: {avg_accuracy}%')
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')


