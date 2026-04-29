# -------------------------------
# 1. Load Libraries
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

# -------------------------------
# 2. Load Dataset (LOCAL FILE)
# -------------------------------
# CSV should be present in the same folder as this script
df = pd.read_csv("gold_prices_1995_2026_feb.csv")

# Convert Date column → datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set Date as index
df.set_index('Date', inplace=True)

# -------------------------------
# 3. Feature Engineering
# -------------------------------

# Lag features
for i in range(1, 6):
    df[f'lag_{i}'] = df['Gold_Price_USD_YFinance'].shift(i)

# Trend feature
df['diff_1'] = df['Gold_Price_USD_YFinance'].diff(1).shift(1)

# Momentum feature
df['diff_2'] = df['Gold_Price_USD_YFinance'].diff(2).shift(1)

# Rolling mean (smoothing)
df['rolling_mean_3'] = df['Gold_Price_USD_YFinance'].rolling(3).mean().shift(1)

# Drop NaN rows
df = df.dropna()

# -------------------------------
# 4. Define X and y
# -------------------------------
X = df[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
        'diff_1', 'diff_2', 'rolling_mean_3']]

y = df['Gold_Price_USD_YFinance']

# -------------------------------
# 5. Train-Test Split (Time-based)
# -------------------------------
split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test  = X[split_index:]

y_train = y[:split_index]
y_test  = y[split_index:]

# -------------------------------
# 6. Train Model
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 7. Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 8. Evaluation
# -------------------------------
rmse = root_mean_squared_error(y_test, y_pred)
print("RMSE:", rmse)

# -------------------------------
# 9. Visualization
# -------------------------------
plt.figure(figsize=(10,5))

plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')

plt.legend()
plt.title("Actual vs Predicted Gold Price")
plt.xlabel("Date")
plt.ylabel("Gold Price (USD)")

plt.show()
