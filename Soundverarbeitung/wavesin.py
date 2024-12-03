import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from fft_selbstgemacht import dft_eigen

# Read the wave file
sample_rate, data = wavfile.read(r"C:\Users\sfz-a\Documents\Soundverarbeitung\test.wav")

# Create a time array
time = np.linspace(0, len(data) / sample_rate, num=len(data))
nu = 1000
print(len(time), len(data))

dft_eigen(data, sample_rate)


# Plot the wave file
plt.figure(figsize=(10, 4))
plt.plot(time[:nu], data[:nu])
plt.title('Wave File as Sine Wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()
