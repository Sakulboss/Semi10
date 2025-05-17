import matplotlib.pyplot as plt

with open(r'results_ff0_1.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.strip('[').strip(']') for line in lines]
    lines = [line.split(', ') for line in lines]
    lines = [line for line in lines if len(line) > 1]
    lines = [[line[0][2:-1], int(line[1][1]), line[2]] for line in lines]

    for i in range(len(lines)):
        for char in range(len(lines[i][2])):
            if not lines[i][2][char] in '0123456789':
                lines[i][2] = int(lines[i][2][:char])
                break

    lines = [[line[0], line[1], int(line[2])] for line in lines]

    for n in range(len(lines)):
        if lines[n][1] == 0:
            lines[n][2] = 100 - lines[n][2]
            lines[n][1] = 1

    lines.sort(key=lambda x: x[0], reverse=True)

# Extract dates and class1 values
dates = [line[0] for line in lines][::-1]
class1_vals = [line[2] for line in lines][::-1]

def plot_values_with_labels(values, labels, label_step=100):
    """
    Plot given values against their labels, showing every label_step-th label.
    :param values: list or array of numeric values (y-axis)
    :param labels: list of string labels (x-axis)
    :param label_step: int, interval between displayed labels
    """
    x_positions = range(len(values))
    plt.plot(x_positions, values)
    # Show every label_step-th label on x-axis
    plt.xticks(ticks=x_positions[::label_step], labels=[labels[i] for i in range(0, len(labels), label_step)], rotation=90)
    plt.xlabel('Labels')
    plt.ylabel('Percentage')
    plt.title(f'Plot with Every {label_step}th Label Shown')
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_values_with_labels(class1_vals, dates)


