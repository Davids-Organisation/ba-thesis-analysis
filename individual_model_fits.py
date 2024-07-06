import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define hyperbolic model
def hyperbolic_func(x, a, b):
    return 1 / (1 + a * x) ** b

# Function to calculate R-squared
def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

# Function to calculate AIC
def calculate_aic(n, rss, k):
    return n * np.log(rss / n) + 2 * k

# Load the data
data = pd.read_csv('treated_data.csv')

# Convert string representations of lists back to actual lists
data['rvSmall'] = data['rvSmall'].apply(lambda x: list(map(float, x.strip('[]').split(','))))
data['rvLarge'] = data['rvLarge'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

# Initialize lists to store results
results = {
    'rvType': [],
    'index': [],
    'model': [],
    'r_value': [],
    'aic': []
}

# Define x values
x = np.array([0.02, 0.1, 0.2, 0.5, 1])

# Function to fit and record results
def fit_and_record(rv_data, rv_type):
    for i, y in enumerate(rv_data):
        if len(set(y)) == 1:  # Check if all values are the same
            print(f"Skipping {rv_type} index {i} because all values are the same.")
            continue

        n = len(y)

        try:
            # Fit hyperbolic model
            popt_hyperbolic, _ = curve_fit(hyperbolic_func, x, y, p0=[0.5, 0.1], maxfev=2000)
            y_pred_hyperbolic = hyperbolic_func(x, *popt_hyperbolic)
            r_squared_hyperbolic = calculate_r_squared(y, y_pred_hyperbolic)
            rss_hyperbolic = np.sum((y - y_pred_hyperbolic) ** 2)
            aic_hyperbolic = calculate_aic(n, rss_hyperbolic, len(popt_hyperbolic))

            # Record hyperbolic results
            results['rvType'].append(rv_type)
            results['index'].append(i)
            results['model'].append('hyperbolic')
            results['r_value'].append(r_squared_hyperbolic)
            results['aic'].append(aic_hyperbolic)

        except RuntimeError:
            print(f"Could not fit hyperbolic model for {rv_type} index {i}. Skipping.")

# Fit and record results for rvSmall and rvLarge
fit_and_record(data['rvSmall'], 'rvSmall')
fit_and_record(data['rvLarge'], 'rvLarge')

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('fitting_results.csv', index=False)

print("Fitting complete. Results saved to 'fitting_results.csv'.")