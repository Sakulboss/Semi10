import argparse
import matplotlib.pyplot as plt
import numpy as np

# run via $ python3 plot.py -f ./cnn.out

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, required=True, help='data file, pairs of comma-separated floats separated by newlines')
args = parser.parse_args()

# Read data from file
filename = args.file
with open(filename, 'r') as file:
    lines = file.readlines()

# Initialize lists to store the data
loss = []
percentage = []

# Parse the data
for line in lines:
    l, p = line.strip().split(',')
    loss.append(float(l))
    percentage.append(float(p))

# Create a plot
fig, ax1 = plt.subplots()

# Plot the loss
color = 'tab:red'
ax1.set_xlabel('Training Batches')
ax1.set_ylabel('Entropy Loss', color=color)
ax1.plot(loss, color=color, alpha=0.3, label='Raw Loss')
ax1.tick_params(axis='y', labelcolor=color)

# Polynomial fitting for loss
degree_loss = 3  # Degree of the polynomial for fitting loss. Adjust as needed.
loss_fit = np.polyfit(range(len(loss)), loss, degree_loss)
p_loss = np.poly1d(loss_fit)
ax1.plot(p_loss(range(len(loss))), color='darkred', label='Fitted Loss')

# Create a second y-axis to plot the percentage
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy percentage', color=color)
ax2.plot(percentage, color=color, alpha=0.3, label='Raw Percentage')
ax2.tick_params(axis='y', labelcolor=color)

# Polynomial fitting for percentage
degree_percentage = 3  # Degree of the polynomial for fitting percentage. Adjust as needed.
percentage_fit = np.polyfit(range(len(percentage)), percentage, degree_percentage)
p_percentage = np.poly1d(percentage_fit)
ax2.plot(p_percentage(range(len(percentage))), color='darkblue', label='Fitted Percentage')

# Add a title, legend, and show the plot
plt.title('Entropy loss vs accuracy% during training')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
fig.tight_layout()
plt.show()
