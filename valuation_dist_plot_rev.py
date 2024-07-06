import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('treated_data.csv')

# Convert string representations of lists back to actual lists
data['rvSmall'] = data['rvSmall'].apply(lambda x: list(map(float, x.strip('[]').split(','))))

# Extract the required data
rvSmall_distance_1 = [row[0] for row in data['rvSmall'][::]]
rvSmall_distance_50 = [row[4] for row in data['rvSmall'][::]]

# Determine the number of bins
num_bins = 30

# Create the first plot with specified figure size
fig1, ax1 = plt.subplots(figsize=(7, 6))
ax1.hist(rvSmall_distance_1, bins=num_bins, density=True)
ax1.set_xlabel('Values')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of choice in small amount condition. – person #1')
plt.savefig('GraphsEPS/distribution_of_choice_small_person_1.eps', format='eps')
plt.savefig('GraphsPDF/distribution_of_choice_small_person_1.pdf', format='pdf')

plt.show()

# Create the second plot with specified figure size
fig2, ax2 = plt.subplots(figsize=(7, 6))
ax2.hist(rvSmall_distance_50, bins=num_bins, density=True)
ax2.set_xlabel('Values')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of choice in small amount condition. – person #50')
plt.savefig('GraphsEPS/distribution_of_choice_small_person_50.eps', format='eps')
plt.savefig('GraphsPDF/distribution_of_choice_small_person_50.pdf', format='pdf')
plt.show()