import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np
from scipy.io import wavfile

def maxelement():
    delindex = yfNew.index(max(yfNew))
    del yfNew[delindex]
    del freq[delindex]

# Loading data (could be from any time-series data source)
data = np.random.randn(1024) # Example data
sample_rate, data = wavfile.read(r"C:\Users\sfz-a\Documents\Soundverarbeitung\test.wav")
# Fourier transform and frequency analysis
N = len(data)
T = N / sample_rate
yf = fft(data)
freq = list(fftfreq(N, T))

yfNew = list(np.abs(yf))

for i in range(100):
    maxelement()

# Plotting
plt.plot(freq, yfNew)
plt.title('Frequency Spectrum')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hz)')
plt.show()