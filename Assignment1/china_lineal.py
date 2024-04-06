# Imports
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Read the file
data = pd.read_csv("chinaOriginal.csv")

# Divide the data into training and testing
X = data.drop(columns=["Effort"])
y = data["Effort"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()               # create the model
linear_model.fit(X_train, y_train)              # train the model using only training data
y_pred_linear = linear_model.predict(X_test)    # predict the effort using testing data

mse_linear = mean_squared_error(y_test, y_pred_linear)  # mean squared error
r2 = r2_score(y_test, y_pred_linear)                    # R^2 score

residuals = y_test - y_pred_linear      # residuals
average_residual = residuals.mean()     # average residual

# Print Metrics
print("Mean Squared Error:", mse_linear)
print("R2 Score:", r2)
print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)
print("Average Residual:", average_residual)

# Visualize the results of the predictions
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_linear, color='blue', label='Actual Values vs Predictions')
# Reference line (perfect prediction)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Actual Values vs Predictions (Linear Model)')
plt.grid(True)
plt.legend()
plt.show()

# Visualize the results for the residuals
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_linear, residuals, color='blue', label='Residuals vs Predictions')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Residuals Diagram')
# Reference line (perfect residuals)
plt.axhline(y=0, color='red', linestyle='--', label='Perfect Residuals')
plt.grid(True)
plt.legend()
plt.show()
