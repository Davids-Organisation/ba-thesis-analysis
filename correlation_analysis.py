import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import linregress
import statsmodels.api as sm
import matplotlib.pyplot as plt

def perform_spearman_test_and_plot(data, col1, col2, label1, label2, plot_title_prefix, age, sex):
    corr, p_value = spearmanr(data[col1], data[col2])
    
    print(f"{plot_title_prefix} - {label2}")
    print(f'Spearman correlation coefficient: {corr}')
    print(f'P-value: {p_value}')
    
    if p_value <= 0.1:
        slope, intercept, r_value, p_value, std_err = linregress(data[col1], data[col2])
        X = pd.DataFrame({
            col2: data[col2],
            'age': age,
            'sex': sex
        })

        y = data[col1]

        # Remove rows with missing values
        X = X.dropna()
        y = y[X.index]

        X = sm.add_constant(X)

        # Fit the model
        model = sm.OLS(y, X).fit()

        # Print the summary
        print(model.summary())

        plt.figure(figsize=(10, 6))
        plt.scatter(data[col1], data[col2], label='Data Points')
        plt.plot(data[col1], intercept + slope * data[col1], 'r', label='Fitted line')
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.title(f'{plot_title_prefix} - {label1} vs {label2}\nR-squared: {r_value**2:.2f}, p-value: {p_value:.4f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.savefig(f'{plot_title_prefix}_{label2}.eps', format='eps')
        plt.show()

# Load the data
data = pd.read_csv('treated_data.csv')

data['binsex'] = data['SD01'].apply(lambda x: 1 if x == 2 else 0)

# List of pairs to test
test_pairs = [
    ('empirical_auc_small', 'agg_acceptance', 'Aggregate Acceptance'),
    ('empirical_auc_large', 'agg_acceptance', 'Aggregate Acceptance'),
    ('empirical_auc_small', 'agg_distance', 'Aggregate Distance'),
    ('empirical_auc_large', 'agg_distance', 'Aggregate Distance'),
    ('empirical_auc_small', 'B201_01', 'Cargo Distance'),
    ('empirical_auc_large', 'B201_01', 'Cargo Distance'),
    ('empirical_auc_small', 'B205_01', 'Tool Distance'),
    ('empirical_auc_large', 'B205_01', 'Tool Distance'),
    ('empirical_auc_small', 'B206_01', 'Garden Distance'),
    ('empirical_auc_large', 'B206_01', 'Garden Distance'),
    ('empirical_auc_small', 'B202', 'Cargo Acceptance'),
    ('empirical_auc_large', 'B202', 'Cargo Acceptance'),
    ('empirical_auc_small', 'B207', 'Tool Acceptance'),
    ('empirical_auc_large', 'B207', 'Tool Acceptance'),
    ('empirical_auc_small', 'B208', 'Garden Acceptance'),
    ('empirical_auc_large', 'B208', 'Garden Acceptance'),
    ('empirical_auc_small', 'B209', 'General Acceptance'),
    ('empirical_auc_large', 'B209', 'General Acceptance'),
]

# Perform tests and plots
for col1, col2, label2 in test_pairs:
    perform_spearman_test_and_plot(data, col1, col2, col1.replace('_', ' ').title(), label2, label2.split()[0], age=data['SD02_01'], sex=data['binsex'])