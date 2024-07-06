import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('treated_data.csv')

# Convert string representations of lists back to actual lists
data['rvSmall'] = data['rvSmall'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

# Extract the data for case 01 (assuming the index is 0 for the first case in Python)
case_number = 26
y_data = [1] + data['rvSmall'].iloc[case_number - 1][:5]
x_data = [0, 0.02, 0.1, 0.2, 0.5, 1]

# Calculate the empirical area under the curve using the trapezoidal rule
current_empirical_auc = np.trapz(y_data, x_data)

# Create the plot
plt.figure(figsize=(7, 6))

plt.plot(x_data, y_data, 's-', linewidth=2, label='Data Points')
plt.fill_between(x_data, y_data, color=[0.9, 0.9, 0.9], alpha=0.5)
plt.xlabel('Distance as Proportion of Maximum')
plt.ylabel('RSV')
plt.title(f'CASE {case_number:02d} – Small – Empirical Area under the Curve')
plt.ylim([0, 1])
plt.grid(True)
plt.text(0.5, 0.8, f'Empirical Area under the Curve: {current_empirical_auc:.4f}', horizontalalignment='center')

# Save the plot as an EPS file with the case number in the file name
file_name = f'auc_individual_plots/CASE_{case_number:02d}_Small_Empirical_Area_under_the_Curve.eps'
file_name_pdf = f'auc_individual_plots/CASE_{case_number:02d}_Small_Empirical_Area_under_the_Curve.pdf'
plt.savefig('GraphsEPS/' + file_name, format='eps')
plt.savefig('GraphsPDF/' + file_name_pdf, format='pdf')
plt.show()