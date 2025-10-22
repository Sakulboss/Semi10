import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

# === Set your audio file path here ===
filename = r"C:\Users\SFZ Rechner\PycharmProjects\Semi10\Sound_processing\_bee_sounds\no_event\output_2025-04-26-05-39-44_359_17.wav"  # Replace with your file name

# Read audio file
samplerate, data = wavfile.read(filename)
print(f"Sample rate: {samplerate} Hz")

# If stereo, take one channel
if data.ndim > 1:
    data = data[:, 0]

# Number of samples
N = len(data)
print(f"Number of samples: {N}")

# Perform Fourier Transform
yf = fft(data)
xf = fftfreq(N, 1 / samplerate)

# Keep only positive frequencies
idx = np.where(xf >= 0)
xf = xf[idx]
yf = np.abs(yf[idx])

n = 50000
xf = xf[:n]
yf = yf[:n]

# Plot frequency spectrum
plt.figure(figsize=(10, 6))
plt.plot(xf, yf)

plt.xlabel("Frequenz (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()
