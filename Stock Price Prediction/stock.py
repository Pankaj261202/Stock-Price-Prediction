import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\py867\OneDrive\Desktop\dataset\Stock.csv")

# Convert string values to float
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

# Drop any NaN values and reset the index
df = df.dropna().reset_index(drop=True)

# Select the 'Close' column as the target variable
y = df['Close']

# Select the features (e.g., 'Open', 'High', 'Low', 'Volume')
X = df[['Open', 'High', 'Low', 'Volume']]

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error (MAE)
mae = metrics.mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

# Plot the predicted vs actual prices
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.title('Stock Price Prediction')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()