import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# Lade die WAV-Datei
sample_rate, data = wavfile.read(r"C:\Users\sfz-a\Documents\Soundverarbeitung\test.wav")

# Berechne die Fourier-Transformation
N = len(data)
T = 1.0 / sample_rate
yf = fft(data)
xf = fftfreq(N, T)[:N//2]

# Plotten des Frequenzspektrums
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()