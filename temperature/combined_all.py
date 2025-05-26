import csv
from datetime import datetime
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd


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


def plot_temperature_in_range(csv_files, start_time=None, end_time=None, save_path=None, smooth_factor=5):
    """
    Plottet Temperaturdaten mit einstellbarer Glättung.

    Parameter:
        csv_files: Liste der CSV-Dateien
        start_time: Startzeit für die Filterung (optional)
        end_time: Endzeit für die Filterung (optional)
        save_path: Pfad zum Speichern des Plots (optional)
        smooth_factor: Größe des Glättungsfensters (je größer, desto stärker die Glättung)
    """
    all_data = []
    all_temps = []
    all_times = []

    for csv_file in csv_files:
        times, temps = read_csv(csv_file, start_time=start_time, end_time=end_time)
        all_data.append((times, temps))
        all_temps.extend(temps)
        all_times.extend(times)

    if not all_temps or not all_times:
        print("Keine Temperatur- oder Zeitdaten in den angegebenen Dateien für den spezifizierten Bereich gefunden.")
        return

    # Berechne Perzentile für robustere Grenzen
    temp_array = np.array(all_temps)
    lower_bound = np.percentile(temp_array, 1)  # 1. Perzentil
    upper_bound = np.percentile(temp_array, 99)  # 99. Perzentil

    # Füge etwas Padding hinzu
    padding = (upper_bound - lower_bound) * 0.05
    y_min = lower_bound - padding
    y_max = upper_bound + padding

    min_time = min(all_times)
    max_time = max(all_times)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mit einstellbarer Glättung
    for i, (times, temps) in enumerate(all_data):
        # Originaldaten mit dünnerer, transparenter Linie
        ax.plot(times, temps, linestyle='-', alpha=0.1,
                color=f'C{i}', linewidth=0.5)

        # Geglättete Daten
        if smooth_factor > 1:
            temps_smooth = pd.Series(temps).rolling(
                window=smooth_factor,
                center=True,
                min_periods=1
            ).mean()
            ax.plot(times, temps_smooth, linestyle='-',
                    label=f'Daten aus {os.path.basename(csv_files[i])}',
                    color=f'C{i}', linewidth=1.5)
        else:
            ax.plot(times, temps, linestyle='-',
                    label=f'Daten aus {os.path.basename(csv_files[i])}',
                    color=f'C{i}', linewidth=1.5)

    ax.set_title(
        f'Temperaturdaten von {min_time.strftime("%Y-%m-%d %H:%M")} bis {max_time.strftime("%Y-%m-%d %H:%M")}\n'
        f'(Glättungsfaktor: {smooth_factor})')
    ax.set_xlabel('Zeit')
    ax.set_ylabel('Temperatur (°C)')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(min_time, max_time)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Kombiniertes Diagramm für den angegebenen Bereich wurde gespeichert als {save_path}")
    else:
        plt.show()


if __name__ == '__main__':
    folder = 'temperature_Marco'
    output_folder = 'diagramme'
    os.makedirs(output_folder, exist_ok=True)

    # Dynamisch alle CSV-Dateien im Ordner einlesen
    csv_files = glob.glob(os.path.join(folder, '*.csv'))
    if not csv_files:
        print(f"Keine CSV-Dateien im Ordner {folder} gefunden")
        exit(1)

    start_time_str = '2025-05-16 00:00:00'
    end_time_str = '2025-05-23 23:59:59'

    start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
    end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')

    save_file = os.path.join(output_folder, f'combined_{folder}_temps.png')

    # Glättungsfaktor als Parameter übergeben (1 = keine Glättung, größere Werte = stärkere Glättung)
    smooth_factor = 500  # Hier können Sie den Wert anpassen

    plot_temperature_in_range(csv_files,
                              start_time=start_time,
                              end_time=end_time,
                              save_path=save_file,
                              smooth_factor=smooth_factor)