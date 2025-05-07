import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import itertools

def read_all_data(file_path):
    data = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dt = datetime.strptime(row[' time'], '%Y-%m-%d %H:%M:%S')
            temp = float(row[' temp'])
            data.append((dt, temp))
    data.sort(key=lambda x: x[0])
    return data

def get_all_days(csv_files):
    unique_days = set()
    for csv_file in csv_files:
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                dt = datetime.strptime(row[' time'], '%Y-%m-%d %H:%M:%S')
                unique_days.add(dt.strftime('%Y-%m-%d'))
    return sorted(unique_days)

def group_times_with_tolerance(list_of_times, tolerance_seconds=10):
    all_times = sorted(set([t for times in list_of_times for t in times]))
    clusters = []
    current_cluster = []
    for t in all_times:
        if not current_cluster:
            current_cluster = [t]
        else:
            if (t - current_cluster[-1]).total_seconds() <= tolerance_seconds:
                current_cluster.append(t)
            else:
                avg_timestamp = avg_datetime(current_cluster)
                clusters.append(avg_timestamp)
                current_cluster = [t]
    if current_cluster:
        avg_timestamp = avg_datetime(current_cluster)
        clusters.append(avg_timestamp)
    return clusters

def avg_datetime(dts):
    total = sum([dt.timestamp() for dt in dts])
    avg_ts = total / len(dts)
    return datetime.fromtimestamp(avg_ts)

def find_closest_time(target, times):
    low, high = 0, len(times) -1
    best_time = None
    best_diff = None
    while low <= high:
        mid = (low + high) // 2
        mid_time = times[mid]
        diff = abs((mid_time - target).total_seconds())
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_time = mid_time
        if mid_time < target:
            low = mid +1
        elif mid_time > target:
            high = mid -1
        else:
            return mid_time
    return best_time

def align_data_on_clusters(data, clusters, tolerance_seconds=10):
    aligned_temps = []
    data_times = [dt for dt, _ in data]
    data_dict = dict(data)
    for cluster_time in clusters:
        closest = find_closest_time(cluster_time, data_times)
        if closest and abs((closest - cluster_time).total_seconds()) <= tolerance_seconds:
            aligned_temps.append(data_dict[closest])
        else:
            aligned_temps.append(None)
    return aligned_temps

def calculate_differences(temps_a, temps_b):
    diffs = []
    for a,b in zip(temps_a, temps_b):
        if a is not None and b is not None:
            diffs.append(b - a)
        else:
            diffs.append(None)
    return diffs

def plot_all_pairwise_differences(csv_files, target_day, save_path=None, tolerance_seconds=10):
    all_data = []
    all_times_lists = []

    for csv_file in csv_files:
        data = read_all_data(csv_file)
        filtered = [(dt,t) for dt,t in data if dt.strftime('%Y-%m-%d') == target_day]
        all_data.append(filtered)
        all_times_lists.append([dt for dt,_ in filtered])

    clusters = group_times_with_tolerance(all_times_lists, tolerance_seconds)
    if not clusters:
        print(f"No timestamps found for day {target_day}. Skipping plot.")
        return

    aligned_temps = [align_data_on_clusters(data, clusters, tolerance_seconds) for data in all_data]

    # Compute pairwise differences
    pair_indices = list(itertools.combinations(range(len(csv_files)), 2))
    diff_results = []
    labels = []
    colors = ['r','g','b','c','m','y','k']  # Extend colors if needed
    for idx, (i,j) in enumerate(pair_indices):
        diff = calculate_differences(aligned_temps[i], aligned_temps[j])
        diff_results.append(diff)
        labels.append(f'Diff {os.path.basename(csv_files[j])} - {os.path.basename(csv_files[i])}')

    # Flatten diffs for y-limit
    flat_diffs = [val for diff in diff_results for val in diff if val is not None]
    if not flat_diffs:
        print(f"No valid temperature differences for day {target_day}. Skipping plot.")
        return
    min_diff = min(flat_diffs)
    max_diff = max(flat_diffs)
    pad = (max_diff - min_diff)*0.1 if max_diff != min_diff else 1

    fig, ax = plt.subplots(figsize=(12,6))

    for idx, diff in enumerate(diff_results):
        ax.plot(clusters, diff, label=labels[idx], linestyle='-', color=colors[idx % len(colors)])

    ax.set_title(f'Pairwise Temperature Differences on {target_day} (±{tolerance_seconds}s tolerance)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature Difference')
    ax.legend()
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(min_diff - pad, max_diff + pad)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved pairwise difference plot for {target_day} as {save_path}")
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
        save_file = os.path.join(output_folder, f'pairwise_differences_{day}.png')
        plot_all_pairwise_differences(csv_files, day, save_path=save_file, tolerance_seconds=10)
