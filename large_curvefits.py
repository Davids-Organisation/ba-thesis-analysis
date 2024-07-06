import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('treated_data.csv')

# Define a function to convert comma-separated strings to lists of floats
def str_to_list(s):
    return list(map(float, s.strip('[]').split(',')))

# Apply the conversion function to the columns to convert strings back to lists
data['rvSmall'] = data['rvSmall'].apply(str_to_list)
data['rvLarge'] = data['rvLarge'].apply(str_to_list)

# Convert to numpy arrays
y_data_small = np.array(data['rvSmall'].tolist())
y_data_large = np.array(data['rvLarge'].tolist())

x = np.array([0.02, 0.1, 0.2, 0.5, 1])

y_small = np.mean(y_data_small, axis=0)
y_large = np.mean(y_data_large, axis=0)

def hyperbolic_func(x, a, b):
    return 1 / (1 + a * x) ** b

def exponential_func(x, a):
    return np.exp(-a * x)

# Initial guess for parameters
initial_params_hyperbolic = [0.5, 0.1]
initial_params_exponential = [0.1]

# Fit the hyperbolic function to the large data
fitted_params_large_hyperbolic, _ = curve_fit(hyperbolic_func, x, y_large, p0=initial_params_hyperbolic)
fitted_hyperbolic_large = hyperbolic_func(x, *fitted_params_large_hyperbolic)

# Fit the exponential function to the large data
fitted_params_large_exponential, _ = curve_fit(exponential_func, x, y_large, p0=initial_params_exponential)
fitted_exponential_large = exponential_func(x, *fitted_params_large_exponential)

# Calculate R-squared for the large data hyperbolic fit
ss_res_large_hyperbolic = np.sum((y_large - fitted_hyperbolic_large) ** 2)
ss_total_large_hyperbolic = np.sum((y_large - np.mean(y_large)) ** 2)
r_squared_large_hyperbolic = 1 - ss_res_large_hyperbolic / ss_total_large_hyperbolic

# Calculate R-squared for the large data exponential fit
ss_res_large_exponential = np.sum((y_large - fitted_exponential_large) ** 2)
ss_total_large_exponential = np.sum((y_large - np.mean(y_large)) ** 2)
r_squared_large_exponential = 1 - ss_res_large_exponential / ss_total_large_exponential

# Calculate AIC for the large data hyperbolic fit
n_large = len(y_large)
k_large_hyperbolic = 2  # Number of parameters in the hyperbolic model
aic_large_hyperbolic = n_large * np.log(ss_res_large_hyperbolic / n_large) + 2 * k_large_hyperbolic

# Calculate AIC for the large data exponential fit
k_large_exponential = 1  # Number of parameters in the exponential model
aic_large_exponential = n_large * np.log(ss_res_large_exponential / n_large) + 2 * k_large_exponential

print(f'AIC for large amount condition hyperbolic fit: {aic_large_hyperbolic:.4f}')
print(f'R-squared for Large amount condition hyperbolic fit: {r_squared_large_hyperbolic:.4f}')
print(f'AIC for large amount condition exponential fit: {aic_large_exponential:.4f}')
print(f'R-squared for Large amount condition exponential fit: {r_squared_large_exponential:.4f}')

# Plot hyperbolic fit for large data and save as EPS
plt.figure(figsize=(7, 6))
plt.plot(x, y_large, 'o', label='Large amount condition - Original')
plt.plot(x, fitted_hyperbolic_large, '-', label='Large amount condition - Fitted (Hyperbolic)')
plt.legend()
plt.xlabel('Distance as Proportion of Maximum')
plt.ylabel('RSV')
plt.title('Hyperbolic Function Fit for Large Data')
plt.ylim([0, 1])
plt.xlim([-0.01, 1.01])
plt.tight_layout()
plt.savefig('GraphsEPS/hyperbolic_fit_large.eps', format='eps')
plt.savefig('GraphsPDF/hyperbolic_fit_large.pdf', format='pdf')
plt.show()

# Plot exponential fit for large data and save as EPS
plt.figure(figsize=(7, 6))
plt.plot(x, y_large, 'o', label='Large amount condition - Original')
plt.plot(x, fitted_exponential_large, '-', label='Large amount condition - Fitted (Exponential)')
plt.legend()
plt.xlabel('Distance as Proportion of Maximum')
plt.ylabel('RSV')
plt.title('Exponential Function Fit for Large Data')
plt.ylim([0, 1])
plt.xlim([-0.01, 1.01])
plt.tight_layout()
plt.savefig('GraphsEPS/exponential_fit_large.eps', format='eps')
plt.savefig('GraphsPDF/exponential_fit_large.pdf', format='pdf')
plt.show()