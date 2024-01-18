
Stock Price Prediction and Analysis, Feel free to adjust it based on your preferences!


This Python script predicts the next day's closing stock price using historical data. It allows users to input their preferred stock 
symbol, visualizes key indicators, makes a prediction using linear regression and assesses prediction accuracy.

# Metrics

![Screenshot 2023-12-03 173051](https://github.com/MoustAhmed/PythonStockPrediction/assets/121663630/232bb5b1-892a-4703-97fc-2221c7fceb8a)

The Mean Absolute Percentage Error (MAPE) is calculated using the formula:

![MAPE Formula](https://latex.codecogs.com/svg.latex?%5Ctext%7BMAPE%7D%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20%5Cleft%7C%20%5Cfrac%7BY_i%20-%20%5Chat%7BY}_i%7D%7BY_i%7D%20%5Cright%7C%20%5Ctimes%20100)

Where:
- `n` is the number of observations.
- `Y_i` is the actual value for observation `i`.
- `hatY_i` is the predicted value for observation `i`.



This formula provides a measure of the average percentage difference between actual and predicted values.


![Figure_1](https://github.com/MoustAhmed/PythonStockPrediction/assets/121663630/c83f5057-f806-42f3-8bd7-1837561956f9)



