import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, shapiro, ttest_rel
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('treated_data.csv')

# Extract data
auc_small = data['empirical_auc_small'][::]
auc_large = data['empirical_auc_large'][::]

# Check for normal distribution
shapiro_small = shapiro(auc_small)
shapiro_large = shapiro(auc_large)

print("Shapiro-Wilk Test for normality:")
print(f"AUC Small: p-value = {shapiro_small.pvalue:.4f}")
print(f"AUC Large: p-value = {shapiro_large.pvalue:.4f}")

# Decide on the test based on normality
if shapiro_small.pvalue > 0.05 and shapiro_large.pvalue > 0.05:
    # Perform paired t-test if both samples are normally distributed
    stat, p = ttest_rel(auc_small, auc_large)
    test_name = "Paired t-test"
else:
    # Perform Wilcoxon signed-rank test if either sample is not normally distributed
    stat, p = wilcoxon(auc_small, auc_large)
    test_name = "Wilcoxon signed-rank test"

print(f"\n{test_name} results:")
print(f'p-value: {p:.4f}')
print(f'Test statistic: {stat}')

# Plot the paired differences
plt.figure(figsize=(10, 6))
paired_differences = auc_small - auc_large
plt.plot(paired_differences, 'o-', linewidth=1.5)
plt.xlabel('Observation')
plt.ylabel('Paired Differences (Small - Large)')
plt.title('Paired Differences Plot')
plt.grid(True)

# Add a horizontal line at y = 0
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

# Add legend and text
plt.legend(['Paired Differences'], loc='best')
plt.text(0.5, 0.1, f'p-value: {p:.4f}', transform=plt.gca().transAxes, backgroundcolor='w')

# Adjust figure properties if needed
plt.tight_layout()
#plt.savefig('paired_differences_plot.eps', format='eps')
plt.show()