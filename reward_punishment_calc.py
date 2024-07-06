import numpy as np
import pandas as pd
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

x = np.array([1, 5, 10, 25, 50])

y_small = np.mean(y_data_small, axis=0)
y_large = np.mean(y_data_large, axis=0)

def calculate_R_and_P(alpha, T, S):
    # Calculate the maximum possible R satisfying the inequality
    R = (T + alpha * S) / (1 + alpha)
    
    # Calculate the minimum possible P satisfying the inequality
    P = (S + alpha * T) / (1 + alpha)
    
    return R, P

def list_to_plot(list, amtc):
    vectorR = []
    vectorP = []
    if(amtc == "small"):
        T = 80
        S = -60
    elif(amtc == "large"):
        T = 1200
        S = -900
    else:
        return
    
    for item in list: 
        R, P = calculate_R_and_P(item, T, S)
        vectorR.append(R)
        vectorP.append(P)

    return vectorR, vectorP

Rs, Ps = list_to_plot(y_small, "small")
Rl, Pl = list_to_plot(y_large, "large")

print(Rs, Ps)
print(Rl, Pl)


# Plot for small data and save as EPS
plt.figure(figsize=(7, 6))
plt.plot(x, Rs, '-', label='Progression of R in small amount condition')
plt.plot(x, Ps, '-', label='Progression of P in small amount condition')
plt.fill_between(x, Rs, 80, color=[0.9, 0.9, 0.9], alpha=0.5)
plt.fill_between(x, -60, Ps, color=[0.9, 0.9, 0.9], alpha=0.5)
plt.legend()
plt.xlabel('Distance in rank of person')
plt.ylabel('Payout amount in €')
plt.title('Progression of Prisoners Dilemma constraints in small ac')
plt.ylim([-60, 80])
plt.xlim([-1, 51])
plt.tight_layout()
plt.savefig('GraphsEPS/pd_constraints_small.eps', format='eps')
plt.savefig('GraphsPDF/pd_constraints_small.pdf', format='pdf')
plt.show()

# Plot for large data and save as EPS
plt.figure(figsize=(7, 6))
plt.plot(x, Rl, '-', label='Progression of R in large amount condition')
plt.plot(x, Pl, '-', label='Progression of P in large amount condition')
plt.fill_between(x, Rl, 1200, color=[0.9, 0.9, 0.9], alpha=0.5)
plt.fill_between(x, -900, Pl, color=[0.9, 0.9, 0.9], alpha=0.5)
plt.legend()
plt.xlabel('Distance in rank of person')
plt.ylabel('Payout amount in €')
plt.title('Progression of Prisoners Dilemma constraints in large ac')
plt.ylim([-900, 1200])
plt.xlim([-1, 51])
plt.tight_layout()
plt.savefig('GraphsEPS/pd_constraints_large.eps', format='eps')
plt.savefig('GraphsPDF/pd_constraints_large.pdf', format='pdf')
plt.show()

