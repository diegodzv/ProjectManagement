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

m, b = np.polyfit(size, effort, 1)              # regression line
effort_predicted = m * size + b                 # predict the effort

residuals = effort - effort_predicted           # residuals
mean_residual = np.mean(np.abs(residuals))      # mean residual

mse = np.mean((effort - effort_predicted)**2)   # mean squared error

# Print the results
print(f"The Linear Regression is: Effort = {m:.2f}*Size + {b:.2f}")
print(f"The Mean Squared Error is: {mse:.2f}")
print(f"The Mean Residual is: {mean_residual:.2f}")

# Plot the data and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(size, effort, color='blue', label='Data')
plt.plot(size, effort_predicted, color='red', label='Linear Regression')
plt.fill_between(size, effort_predicted - mean_residual, effort_predicted + mean_residual,
                 color='green', alpha=0.2, label='Mean Residual Region')
plt.title('Size vs Effort')
plt.xlabel('Size')    
plt.ylabel('Effort')
plt.grid(True)
plt.legend()
plt.show()
