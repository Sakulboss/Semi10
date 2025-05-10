import csv
from datetime import datetime
import matplotlib.pyplot as plt
import os


def read_csv(file_path):
    """
    Reads CSV file and returns lists of datetime objects and temperature floats.
    """
    times = []
    temps = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dt = datetime.strptime(row[' time'], '%Y-%m-%d %H:%M:%S')
            times.append(dt)
            temps.append(float(row[' temp']))
    return times, temps


def plot_all_temperature_data(csv_files, save_path=None):
    all_data = []
    all_temps = []
    all_times = []

    for csv_file in csv_files:
        times, temps = read_csv(csv_file)
        all_data.append((times, temps))
        all_temps.extend(temps)
        all_times.extend(times)

    if not all_temps or not all_times:
        print("No temperature or time data found in the provided files.")
        return

    min_temp = min(all_temps)
    max_temp = max(all_temps)
    y_max = max_temp + (max_temp - min_temp) * 0.1 if max_temp != min_temp else max_temp + 1
    y_min = min_temp - (max_temp - min_temp) * 0.1 if max_temp != min_temp else min_temp - 1

    min_time = min(all_times)
    max_time = max(all_times)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all temperatures from different CSV files in one plot
    for i, (times, temps) in enumerate(all_data):
        ax.plot(times, temps, linestyle='-', label=f'Data from {os.path.basename(csv_files[i])}')

    ax.set_title('Combined Temperature Data from All CSV Files')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(min_time, max_time)
    ax.legend()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)  # Close to free memory, do not show the plot
        print(f"Saved combined plot as {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    folder = 'temperature_FF'
    output_folder = 'diagramme'
    os.makedirs(output_folder, exist_ok=True)

    csv_files = [
        os.path.join(folder, '27f3c5356461.csv'),
        os.path.join(folder, '47cdfb356461.csv'),
        os.path.join(folder, 'e2e4c5356461.csv'),
        os.path.join(folder, 'ede8c5356461.csv')
    ]

    save_file = os.path.join(output_folder, 'combined_temps_all_days.png')
    plot_all_temperature_data(csv_files, save_path=save_file)