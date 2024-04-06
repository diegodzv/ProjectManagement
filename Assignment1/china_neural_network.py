# Imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, median_absolute_error

from sklearn.neural_network import MLPRegressor

# Read the file
data = pd.read_csv("chinaOriginal.csv")

# Divide the data into training and testing
X = data.drop(columns=["Effort"])
y = data["Effort"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Multi-layer Perceptron regressor Model
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)

mlp_model.fit(X_train, y_train)         # train the model
y_pred_mlp = mlp_model.predict(X_test)  # predict the effort

# Metrics
mse_mlp = mean_squared_error(y_test, y_pred_mlp)    # mean squared error
r2_mlp = r2_score(y_test, y_pred_mlp)               # R^2 score

residuals = y_test - y_pred_mlp         # residuals
average_residual = residuals.mean()     # average residual

# Print Metrics
print("Mean Squared Error (RMSE):", mse_mlp)
print("R2 Score (MLPRegressor):", r2_mlp)
print("Average Residual:", average_residual)

# Visualize the results of the predictions
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_mlp, color='blue', label='Actual Values vs Predictions')
# Reference line (perfect prediction)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Actual Values vs Predictions (Neural Network)')
plt.grid(True)
plt.legend()
plt.show()

# Visualize the results for the residuals
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_mlp, residuals, color='blue', label='Residuals vs Predictions')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Residuals Diagram (MLP)')
# Reference line (perfect residuals)
plt.axhline(y=0, color='red', linestyle='--', label='Perfect Residuals')
plt.grid(True)
plt.legend()
plt.show()
