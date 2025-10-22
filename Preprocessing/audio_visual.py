import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Replace 'your_file.wav' with the path to your WAV file
wav_file = r'C:\Users\SFZ Rechner\PycharmProjects\Semi10\Sound_processing\_bee_sounds\no_event\output_2025-04-26-05-39-44_359_17.wav'

# Read the WAV file
sample_rate, data = wavfile.read(wav_file)

# Check if stereo and convert to mono if necessary
if len(data.shape) == 2:
    data = data[2052000:2100000, 0]  # Take only the first channel
print(max(data))
print(min(data))
#print(list(data).index(-0.5)) #321
# Create time axis in seconds
time = np.linspace(0, len(data) / sample_rate, num=len(data))

# Plot the waveform
plt.figure(figsize=(10, 4))
plt.plot(time, data, color='blue')
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()