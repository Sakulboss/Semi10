import csv
from datetime import datetime
import matplotlib.pyplot as plt
import os
import glob


def read_csv(file_path, start_time=None, end_time=None):
    """
    Reads CSV file and returns lists of datetime objects and temperature floats,
    filtered by optional start_time and end_time datetime bounds.
    """
    times = []
    temps = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dt = datetime.strptime(row[' time'], '%Y-%m-%d %H:%M:%S')
            if (start_time is None or dt >= start_time) and (end_time is None or dt <= end_time):
                times.append(dt)
                temps.append(float(row[' temp']))
    return times, temps


def plot_temperature_in_range(csv_files, start_time=None, end_time=None, save_path=None):
    all_data = []
    all_temps = []
    all_times = []

    for csv_file in csv_files:
        times, temps = read_csv(csv_file, start_time=start_time, end_time=end_time)
        all_data.append((times, temps))
        all_temps.extend(temps)
        all_times.extend(times)

    if not all_temps or not all_times:
        print("No temperature or time data found in the provided files for the specified range.")
        return

    min_temp = min(all_temps)
    max_temp = max(all_temps)
    y_max = max_temp + (max_temp - min_temp) * 0.1 if max_temp != min_temp else max_temp + 1
    y_min = min_temp - (max_temp - min_temp) * 0.1 if max_temp != min_temp else min_temp - 1

    min_time = min(all_times)
    max_time = max(all_times)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all temperatures from different CSV files in one plot
    for i, (times, temps) in enumerate(all_data):
        ax.plot(times, temps, linestyle='-', label=f'Data from {os.path.basename(csv_files[i])}')

    ax.set_title(f'Temperature Data from {min_time.strftime("%Y-%m-%d %H:%M")} to {max_time.strftime("%Y-%m-%d %H:%M")}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature')
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(min_time, max_time)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)  # Close to free memory, do not show the plot
        print(f"Saved combined plot for specified range as {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    folder = 'temperature_FF'
    output_folder = 'diagramme'
    os.makedirs(output_folder, exist_ok=True)

    # Dynamisch alle CSV-Dateien im Ordner einlesen
    csv_files = glob.glob(os.path.join(folder, '*.csv'))
    if not csv_files:
        print(f"No CSV files found in folder {folder}")
        exit(1)

    # Beispiel-Zeitraum (kann auch auf None gesetzt werden für alle Daten)
    start_time_str = '2025-05-09 00:00:00'  # Beispiel: Startdatum
    end_time_str = '2025-05-16 23:59:59'    # Beispiel: Enddatum

    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')

    save_file = os.path.join(output_folder, f'combined_temps_{start_time_str[:10]}_to_{end_time_str[:10]}.png')
    plot_temperature_in_range(csv_files, start_time=start_time, end_time=end_time, save_path=save_file)