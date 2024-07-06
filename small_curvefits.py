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

# Fit the hyperbolic function to the small data
fitted_params_small_hyperbolic, _ = curve_fit(hyperbolic_func, x, y_small, p0=initial_params_hyperbolic)
fitted_hyperbolic_small = hyperbolic_func(x, *fitted_params_small_hyperbolic)

# Fit the exponential function to the small data
fitted_params_small_exponential, _ = curve_fit(exponential_func, x, y_small, p0=initial_params_exponential)
fitted_exponential_small = exponential_func(x, *fitted_params_small_exponential)

# Calculate R-squared for the small data hyperbolic fit
ss_res_small_hyperbolic = np.sum((y_small - fitted_hyperbolic_small) ** 2)
ss_total_small_hyperbolic = np.sum((y_small - np.mean(y_small)) ** 2)
r_squared_small_hyperbolic = 1 - ss_res_small_hyperbolic / ss_total_small_hyperbolic

# Calculate R-squared for the small data exponential fit
ss_res_small_exponential = np.sum((y_small - fitted_exponential_small) ** 2)
ss_total_small_exponential = np.sum((y_small - np.mean(y_small)) ** 2)
r_squared_small_exponential = 1 - ss_res_small_exponential / ss_total_small_exponential

# Calculate AIC for the small data hyperbolic fit
n_small = len(y_small)
k_small_hyperbolic = 2  # Number of parameters in the hyperbolic model
aic_small_hyperbolic = n_small * np.log(ss_res_small_hyperbolic / n_small) + 2 * k_small_hyperbolic

# Calculate AIC for the small data exponential fit
k_small_exponential = 1  # Number of parameters in the exponential model
aic_small_exponential = n_small * np.log(ss_res_small_exponential / n_small) + 2 * k_small_exponential

print(f'AIC for small amount condition hyperbolic fit: {aic_small_hyperbolic:.4f}')
print(f'R-squared for Small amount condition hyperbolic fit: {r_squared_small_hyperbolic:.4f}')
print(f'AIC for small amount condition exponential fit: {aic_small_exponential:.4f}')
print(f'R-squared for Small amount condition exponential fit: {r_squared_small_exponential:.4f}')

# Plot hyperbolic fit for small data and save as EPS
plt.figure(figsize=(7, 6))
plt.plot(x, y_small, 'o', label='Small amount condition - Original')
plt.plot(x, fitted_hyperbolic_small, '-', label='Small amount condition - Fitted (Hyperbolic)')
plt.legend()
plt.xlabel('Distance as Proportion of Maximum')
plt.ylabel('RSV')
plt.title('Hyperbolic Function Fit for Small Data')
plt.ylim([0, 1])
plt.xlim([-0.01, 1.01])
plt.tight_layout()
plt.savefig('GraphsEPS/hyperbolic_fit_small.eps', format='eps')
plt.savefig('GraphsPDF/hyperbolic_fit_small.pdf', format='pdf')
plt.show()

# Plot exponential fit for small data and save as EPS
plt.figure(figsize=(7, 6))
plt.plot(x, y_small, 'o', label='Small amount condition - Original')
plt.plot(x, fitted_exponential_small, '-', label='Small amount condition - Fitted (Exponential)')
plt.legend()
plt.xlabel('Distance as Proportion of Maximum')
plt.ylabel('RSV')
plt.title('Exponential Function Fit for Small Data')
plt.ylim([0, 1])
plt.xlim([-0.01, 1.01])
plt.tight_layout()
plt.savefig('GraphsEPS/exponential_fit_small.eps', format='eps')
plt.savefig('GraphsPDF/exponential_fit_small.pdf', format='pdf')
plt.show()