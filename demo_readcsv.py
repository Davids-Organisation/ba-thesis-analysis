###Perform the "demo_decision_phase.html" in the "Combinatioin_Discounting_Task" first and store the resulting csv "demo_results.csv" into this python project folder.


import pandas as pd 
import numpy as np
import os

data = pd.read_csv('demo_results.csv')


#data['agg_distance'] = data[['B201_01', 'B205_01', 'B206_01']].apply(lambda row: row.sum(), axis=1)
#data['agg_acceptance'] = data[['B202', 'B207', 'B208', 'B209']].apply(lambda row: row.sum(), axis=1)

data['smalls'] = data[['small1', 'small5', 'small10', 'small25', 'small50']].values.tolist()
data['larges'] = data[['large1', 'large5', 'large10', 'large25', 'large50']].values.tolist()

data['rvSmall'] = [(np.array(row) + 60) / 80 for row in data['smalls']]
data['rvLarge'] = [(np.array(row) + 900) / 1200 for row in data['larges']]

def calculate_empirical_auc(x, y):
    return np.trapz(y, x)

x_small_full = [0, 1/50, 5/50, 10/50, 25/50, 1]
x_small_short = [0, 1/10, 5/10, 1]
x_small_long = [0, 1/5, 1/2, 1]

x_large_full = [0, 1/50, 5/50, 10/50, 25/50, 1]
x_large_short = [0, 1/10, 5/10, 1]
x_large_long = [0, 1/5, 1/2, 1]

data['empirical_auc_small'] = [calculate_empirical_auc(x_small_full, [1] + list(rv)) for rv in data['rvSmall']]
data['empirical_auc_small_short'] = [calculate_empirical_auc(x_small_short, [1] + list(rv[:3])) for rv in data['rvSmall']]
data['empirical_auc_small_long'] = [calculate_empirical_auc(x_small_long, [1] + list(rv[2:5])) for rv in data['rvSmall']]

data['empirical_auc_large'] = [calculate_empirical_auc(x_large_full, [1] + list(rv)) for rv in data['rvLarge']]
data['empirical_auc_large_short'] = [calculate_empirical_auc(x_large_short, [1] + list(rv[:3])) for rv in data['rvLarge']]
data['empirical_auc_large_long'] = [calculate_empirical_auc(x_large_long, [1] + list(rv[2:5])) for rv in data['rvLarge']]


# Convert lists to comma-separated strings
data['rvSmall'] = data['rvSmall'].apply(lambda x: '[' + ','.join(map(str, x)) + ']')
data['rvLarge'] = data['rvLarge'].apply(lambda x: '[' + ','.join(map(str, x)) + ']')

# Print results
#for i, auc in enumerate(data['empirical_auc_small']):
#    print(f'Empirical AUC for small (full) i = {i}: {auc:.4f}')
#for i, auc in enumerate(data['empirical_auc_small_short']):
#    print(f'Empirical AUC for small (short) i = {i}: {auc:.4f}')
#for i, auc in enumerate(data['empirical_auc_small_long']):
#    print(f'Empirical AUC for small (long) i = {i}: {auc:.4f}')
#for i, auc in enumerate(data['empirical_auc_large']):
#    print(f'Empirical AUC for large (full) i = {i}: {auc:.4f}')
#for i, auc in enumerate(data['empirical_auc_large_short']):
#    print(f'Empirical AUC for large (short) i = {i}: {auc:.4f}')
#for i, auc in enumerate(data['empirical_auc_large_long']):
#    print(f'Empirical AUC for large (long) i = {i}: {auc:.4f}')

output_file = 'treated_data.csv'
data.to_csv(output_file, index=False)
print(f'Data has been exported to {output_file}')

folders = ["GraphsEPS", "GraphsPDF"]

# Create the folder
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Folder '{folder}' created successfully!")
    else:
        print(f"Folder '{folder}' already exists.")