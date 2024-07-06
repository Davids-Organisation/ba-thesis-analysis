import pandas as pd 
import numpy as np

# Load the data
data = pd.read_csv('treated_data.csv')

# Convert age column to numeric
data["SD02_01"] = pd.to_numeric(data["SD02_01"], errors='coerce')

# Number of cases
num_cases = len(data["SD01"])

# Mean age
mean_age = np.mean(data["SD02_01"])

# Standard deviation of age
std_age = np.std(data["SD02_01"])

# Range of age
range_age = data["SD02_01"].max() - data["SD02_01"].min()

# Gender counts
women = data["SD01"].value_counts().get(1, 0)
men = data["SD01"].value_counts().get(2, 0)
other = data["SD01"].value_counts().get(3, 0)

# Print the results
print(f'Number of cases: {num_cases}')
print(f'Mean Age: {mean_age}')
print(f'Standard Deviation of Age: {std_age}')
print(f'Range of Age: {range_age}')
print(f'Women: {women}')
print(f'Men: {men}')
print(f'Other: {other}')