# Imports
import arff
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
with open('./albretch.arff', 'r') as f:
    data = arff.load(f)

# Convert the data to numpy arrays
size = np.array([row[0] for row in data['data']])
effort = np.array([row[1] for row in data['data']])

coefficients = np.polyfit(size, effort, 2)      # polynomial regression

# Function to calculate the predicted values
def polyval(coefficients, x):
    return coefficients[0] * x**2 + coefficients[1] * x + coefficients[2]

effort_predicted = polyval(coefficients, size)  # predict the effort

residuals = effort - effort_predicted           # residuals
mean_residual = np.mean(np.abs(residuals))      # mean residual

mse = np.mean((effort - effort_predicted)**2)   # mean squared error

# Equation of the polynomial regression
equation = f"Effort = {coefficients[0]:.2f} * Size^2 + {coefficients[1]:.2f} * Size + {coefficients[2]:.2f}"

# Print the results
print("Equation of the Polynomial Regression:", equation)
print(f"The Mean Squared Error is: {mse:.2f}")
print(f"The Mean Residual is: ", mean_residual)

# Plot the data and the polynomial regression
plt.figure(figsize=(8, 6))
plt.scatter(size, effort, color='blue', label='Data')
sorted_indices = np.argsort(size)
plt.plot(size[sorted_indices], effort_predicted[sorted_indices], color='red', label='Polynomial Regression')
plt.fill_between(size[sorted_indices], effort_predicted[sorted_indices] - mean_residual,
                 effort_predicted[sorted_indices] + mean_residual, color='green',
                 alpha=0.2, label='Mean Residual Region')
plt.title('Size vs Effort')
plt.xlabel('Size')
plt.ylabel('Effort')
plt.grid(True)
plt.legend()
plt.show()
