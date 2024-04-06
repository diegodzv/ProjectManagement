# Imports
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.svm import SVR

# Read the file
data = pd.read_csv("chinaOriginal.csv")

# Divide the data into training and testing
X = data.drop(columns=["Effort"])
y = data["Effort"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the SVR model
svr_model = SVR(kernel='linear')

svr_model.fit(X_train, y_train)         # train the model
y_pred_svr = svr_model.predict(X_test)  # predict the effort

# Metrics
mse_svr = mean_squared_error(y_test, y_pred_svr)    # mean squared error
r2_svr = r2_score(y_test, y_pred_svr)               # R^2 score

residuals_svr = y_test - y_pred_svr         # residuals
average_residual_svr = residuals_svr.mean() # mean residuals

# Print Metrics
print("Mean Squared Error (SVR):", mse_svr)
print("R2 Score (SVR):", r2_svr)
print("Average Residual (SVR):", average_residual_svr)

# Visualize the results of the predictions
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_svr, color='blue', label='Actual Values vs Predictions')
# Reference line (perfect prediction)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Perfect Predictions')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Actual Values vs Predictions (Support Vector Regression)')
plt.grid(True)
plt.legend()
plt.show()

# Visualize the results for the residuals
plt.figure(figsize=(6, 6))
plt.scatter(y_pred_svr, residuals_svr, color='blue', label='Residuals vs Predictions')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Residuals Diagram (SVR)')
# Reference line (perfect residuals)
plt.axhline(y=0, color='red', linestyle='--', label='Perfect Residuals')
plt.grid(True)
plt.legend()
plt.show()
