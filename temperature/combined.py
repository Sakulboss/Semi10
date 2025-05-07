import csv
from datetime import datetime
import matplotlib.pyplot as plt
import os


def read_and_filter_csv(file_path, target_day=None):
    """
    Reads CSV file and filters rows for the target_day (YYYY-MM-DD).
    If target_day is None, returns all data.
    Returns lists of datetime objects and temperature floats.
    """
    times = []
    temps = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dt = datetime.strptime(row[' time'], '%Y-%m-%d %H:%M:%S')
            if target_day is None or dt.strftime('%Y-%m-%d') == target_day:
                times.append(dt)
                temps.append(float(row[' temp']))
    return times, temps


def get_all_days(csv_files):
    """
    Scans all CSV files and returns a sorted list of unique days (YYYY-MM-DD) found in all files.
    """
    unique_days = set()
    for csv_file in csv_files:
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dt = datetime.strptime(row[' time'], '%Y-%m-%d %H:%M:%S')
                unique_days.add(dt.strftime('%Y-%m-%d'))
    return sorted(unique_days)


def plot_temperature_for_day(csv_files, target_day, save_path=None):
    all_data = []
    all_temps = []
    all_times = []

    for csv_file in csv_files:
        times, temps = read_and_filter_csv(csv_file, target_day)
        all_data.append((times, temps))
        all_temps.extend(temps)
        all_times.extend(times)

    if not all_temps or not all_times:
        print(f"No temperature or time data found for the day {target_day} in the provided files.")
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

    ax.set_title(f'Temperature Data for {target_day} from 4 CSV files')
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
        print(f"Saved plot for {target_day} as {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    folder = 'temperature_data'
    output_folder = 'diagramme'
    os.makedirs(output_folder, exist_ok=True)

    csv_files = [
        os.path.join(folder, '27f3c5356461.csv'),
        os.path.join(folder, '47cdfb356461.csv'),
        os.path.join(folder, 'e2e4c5356461.csv'),
        os.path.join(folder, 'ede8c5356461.csv')
    ]

    all_days = get_all_days(csv_files)
    print(f"Found {len(all_days)} unique days in data.")

    for day in all_days:
        save_file = os.path.join(output_folder, f'combined_temps_{day}.png')
        plot_temperature_for_day(csv_files, day, save_path=save_file)
