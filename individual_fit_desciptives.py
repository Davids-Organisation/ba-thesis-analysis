import pandas as pd

# Load the fitting results
results_df = pd.read_csv('fitting_results.csv')

# Filter the DataFrame
hyperbolic_rvSmall = results_df[(results_df['model'] == 'hyperbolic') & (results_df['rvType'] == 'rvSmall')]
hyperbolic_rvLarge = results_df[(results_df['model'] == 'hyperbolic') & (results_df['rvType'] == 'rvLarge')]

# Calculate the mean R-squared value
mean_r_squared_hyperbolic_rvSmall = hyperbolic_rvSmall['r_value'].mean()
mean_r_squared_hyperbolic_rvLarge = hyperbolic_rvLarge['r_value'].mean()

mean_aic_hyperbolic_rvSmall = hyperbolic_rvSmall['aic'].mean()
mean_aic_hyperbolic_rvLarge = hyperbolic_rvLarge['aic'].mean()

print(f'Mean R-squared value of hyperbolic fit on rvSmall: {mean_r_squared_hyperbolic_rvSmall:.4f}')
print(f'Mean R-squared value of hyperbolic fit on rvLarge: {mean_r_squared_hyperbolic_rvLarge:.4f}')
print(f'Mean AIC value of hyperbolic fit on rvSmall: {mean_aic_hyperbolic_rvSmall:.4f}')
print(f'Mean AIC value of hyperbolic fit on rvLarge: {mean_aic_hyperbolic_rvLarge:.4f}')