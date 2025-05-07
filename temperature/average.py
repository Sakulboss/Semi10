import csv
from datetime import datetime
import matplotlib.pyplot as plt
import os

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
    low, high = 0, len(times) - 1
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
            low = mid + 1
        elif mid_time > target:
            high = mid - 1
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

def calculate_diff_to_avg(aligned_temps):
    """
    Für jeden Zeitpunkt den Durchschnitt der vorhandenen Werte berechnen
    und für jede Datei die Differenz zum Durchschnitt.
    """
    n = len(aligned_temps)
    length = len(aligned_temps[0]) if n > 0 else 0

    diffs = []
    # Durchschnitt pro Zeitpunkt berechnen (nur existierende Werte)
    avg_temps = []
    for i in range(length):
        vals = [aligned_temps[d][i] for d in range(n) if aligned_temps[d][i] is not None]
        avg = sum(vals) / len(vals) if vals else None
        avg_temps.append(avg)

    # Differenzen zu Durchschnitt für jede Datei
    for d in range(n):
        diff_ds = []
        for i in range(length):
            if aligned_temps[d][i] is None or avg_temps[i] is None:
                diff_ds.append(None)
            else:
                diff_ds.append(aligned_temps[d][i] - avg_temps[i])
        diffs.append(diff_ds)

    return diffs

def plot_diff_to_average(csv_files, target_day, save_path=None, tolerance_seconds=10):
    all_data = []
    all_times_lists = []

    for csv_file in csv_files:
        data = read_all_data(csv_file)
        filtered = [(dt,t) for dt,t in data if dt.strftime('%Y-%m-%d') == target_day]
        all_data.append(filtered)
        all_times_lists.append([dt for dt,_ in filtered])

    clusters = group_times_with_tolerance(all_times_lists, tolerance_seconds)
    if not clusters:
        print(f"Keine Zeitstempel für den Tag {target_day} gefunden. Diagramm wird übersprungen.")
        return

    aligned_temps = [align_data_on_clusters(data, clusters, tolerance_seconds) for data in all_data]

    diffs = calculate_diff_to_avg(aligned_temps)

    flat_diffs = [val for ds in diffs for val in ds if val is not None]
    if not flat_diffs:
        print(f"Keine gültigen Differenzdaten für den Tag {target_day}. Diagramm wird übersprungen.")
        return

    min_diff = min(flat_diffs)
    max_diff = max(flat_diffs)
    padding = (max_diff - min_diff) * 0.1 if max_diff != min_diff else 1

    colors = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, diff_ds in enumerate(diffs):
        label = f'Differenz {os.path.basename(csv_files[idx])} zum Durchschnitt'
        ax.plot(clusters, diff_ds, label=label, linestyle='-', color=colors[idx % len(colors)])

    ax.set_title(f'Temperaturdifferenzen zum Durchschnitt am {target_day} (±{tolerance_seconds}s Toleranz)')
    ax.set_xlabel('Zeit')
    ax.set_ylabel('Temperaturdifferenz')
    ax.grid(True)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(min_diff - padding, max_diff + padding)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Diagramm für {target_day} gespeichert: {save_path}")
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
    print(f"{len(all_days)} Tage mit Daten gefunden.")

    for day in all_days:
        save_file = os.path.join(output_folder, f'average_differences_{day}.png')
        plot_diff_to_average(csv_files, day, save_path=save_file, tolerance_seconds=10)
