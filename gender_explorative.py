import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('treated_data.csv')

# Create a binary column for sex
data['binsex'] = data['SD01'].apply(lambda x: 1 if x == 2 else 0)

# Extract AUC values for men (binsex=1) and non-men (binsex=0)
auc_small_men = data.loc[data['binsex'] == 1, 'empirical_auc_small']
auc_small_non_men = data.loc[data['binsex'] == 0, 'empirical_auc_small']

auc_large_men = data.loc[data['binsex'] == 1, 'empirical_auc_large']
auc_large_non_men = data.loc[data['binsex'] == 0, 'empirical_auc_large']

# Check for normal distribution
shapiro_men_small = shapiro(auc_small_men)
shapiro_non_men_small = shapiro(auc_small_non_men)
shapiro_men_large = shapiro(auc_large_men)
shapiro_non_men_large = shapiro(auc_large_non_men)

print("Shapiro-Wilk Test for normality:")
print(f"AUC Small Men: p-value = {shapiro_men_small.pvalue:.4f}")
print(f"AUC Small Non-Men: p-value = {shapiro_non_men_small.pvalue:.4f}")
print(f"AUC Large Men: p-value = {shapiro_men_large.pvalue:.4f}")
print(f"AUC Large Non-Men: p-value = {shapiro_non_men_large.pvalue:.4f}")

# Decide on the test based on normality for small AUC
if shapiro_men_small.pvalue > 0.05 and shapiro_non_men_small.pvalue > 0.05:
    # Perform t-test for small AUC if normally distributed
    t_stat_small, p_small = ttest_ind(auc_small_men, auc_small_non_men)
    df_small = len(auc_small_men) + len(auc_small_non_men) - 2
    print("\nT-test for AUC Small:")
    print(f'T-test statistic: {t_stat_small:.4f}, Degrees of freedom: {df_small}, p-value: {p_small:.4f}')
else:
    # Perform Mann-Whitney U test for small AUC if not normally distributed
    u_stat_small, p_small = mannwhitneyu(auc_small_men, auc_small_non_men)
    print("\nMann-Whitney U test for AUC Small:")
    print(f'U test statistic: {u_stat_small:.4f}, p-value: {p_small:.4f}')

# Decide on the test based on normality for large AUC
if shapiro_men_large.pvalue > 0.05 and shapiro_non_men_large.pvalue > 0.05:
    # Perform t-test for large AUC if normally distributed
    t_stat_large, p_large = ttest_ind(auc_large_men, auc_large_non_men)
    df_large = len(auc_large_men) + len(auc_large_non_men) - 2
    print("\nT-test for AUC Large:")
    print(f'T-test statistic: {t_stat_large:.4f}, Degrees of freedom: {df_large}, p-value: {p_large:.4f}')
else:
    # Perform Mann-Whitney U test for large AUC if not normally distributed
    u_stat_large, p_large = mannwhitneyu(auc_large_men, auc_large_non_men)
    print("\nMann-Whitney U test for AUC Large:")
    print(f'U test statistic: {u_stat_large:.4f}, p-value: {p_large:.4f}')

# Plot the paired differences for small AUC
plt.figure(figsize=(10, 6))
paired_differences_small = auc_small_men.values - np.resize(auc_small_non_men.values, auc_small_men.shape)
plt.plot(paired_differences_small, 'o-', linewidth=1.5)
plt.xlabel('Observation')
plt.ylabel('Paired Differences (men - non-men)')
plt.title('Paired Differences Plot for Small AUC')
plt.grid(True)

# Add a horizontal line at y = 0
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

# Add legend and text
plt.legend(['Paired Differences'], loc='best')
plt.text(0.5, 0.1, f'p-value: {p_small:.4f}', transform=plt.gca().transAxes, backgroundcolor='w')

# Adjust figure properties if needed
plt.tight_layout()
#plt.savefig('paired_differences_plot_small.eps', format='eps')
plt.show()

# Plot the paired differences for large AUC
plt.figure(figsize=(10, 6))
paired_differences_large = auc_large_men.values - np.resize(auc_large_non_men.values, auc_large_men.shape)
plt.plot(paired_differences_large, 'o-', linewidth=1.5)
plt.xlabel('Observation')
plt.ylabel('Paired Differences (men - non-men)')
plt.title('Paired Differences Plot for Large AUC')
plt.grid(True)

# Add a horizontal line at y = 0
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)

# Add legend and text
plt.legend(['Paired Differences'], loc='best')
plt.text(0.5, 0.1, f'p-value: {p_large:.4f}', transform=plt.gca().transAxes, backgroundcolor='w')

# Adjust figure properties if needed
plt.tight_layout()
#plt.savefig('paired_differences_plot_large.eps', format='eps')
plt.show()

# Plot distributions of the four groups on one plane
plt.figure(figsize=(14, 8))
sns.kdeplot(auc_small_men, label='AUC Small Men', fill=True)
sns.kdeplot(auc_small_non_men, label='AUC Small Non-Men', fill=True)
sns.kdeplot(auc_large_men, label='AUC Large Men', fill=True)
sns.kdeplot(auc_large_non_men, label='AUC Large Non-Men', fill=True)
plt.xlabel('AUC Values')
plt.ylabel('Density')
plt.title('Distributions of AUC Values')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig('auc_distributions.eps', format='eps')
plt.show()