import matplotlib.pyplot as plt
import re


def read_mse_values_from_file(file_path):
    mse_values = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                # Suche nach Prozentzahlen in der Zeile
                matches = re.findall(r'(\d+\.\d+)%', line)
                for match in matches:
                    try:
                        mse_value = float(match)  # Umwandlung in float
                        mse_values.append(mse_value)
                    except ValueError:
                        continue
    return mse_values


def plot_mse_parabola(mse_values):
    epochs = list(range(1, len(mse_values) + 1))  # Erstelle eine Liste von Epochen
    plt.figure(figsize=(10, 6))

    # Parabel plotten
    plt.plot(epochs, mse_values, marker='o', linestyle='-', color='b')
    plt.title('Mean Squared Error über Epochen')
    plt.xlabel('Epochen')
    plt.ylabel('Mean Squared Error (%)')
    plt.ylim(0, max(mse_values) * 1.1)  # y-Achse etwas größer als der maximale MSE-Wert
    plt.grid()
    plt.xticks(epochs)  # Epochen als x-Achsen-Ticks
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = 'data.txt'  # Datei mit den Zeilen
    mse_values = read_mse_values_from_file(file_path)

    if mse_values:
        plot_mse_parabola(mse_values)
import torch

def check_gpu():
    if torch.cuda.is_available():
        print("[PASS] ROCm-kompatible GPU erkannt.")
        print(f"Gefundene GPU(s): {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f" - Gerät {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Keine gültigen MSE-Werte in der Datei gefunden.")
        print("[FAIL] Keine GPU erkannt oder ROCm nicht aktiv.")

if __name__ == "__main__":
    check_gpu()