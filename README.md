# house-price-prediction
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset (House size in square feet)
X = np.array([[500],
              [800],
              [1000],
              [1200],
              [1500],
              [1800]])

# House price (in thousands)
y = np.array([150, 200, 250, 300, 350, 400])

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict price for new house
size = np.array([[1300]])
predicted_price = model.predict(size)

print("Predicted House Price:", predicted_price)

# Plot dataset
plt.scatter(X, y, color="blue", label="Actual Data")

# Plot regression line
plt.plot(X, model.predict(X), color="red", label="Regression Line")

plt.xlabel("House Size (sq ft)")
plt.ylabel("Price (in thousands)")
plt.title("House Price Prediction using Linear Regression")
plt.legend()

plt.show()
