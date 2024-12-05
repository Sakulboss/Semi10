import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from pathfinder import soundpath

y, sr = librosa.load(soundpath('test.wav'))
plt.plot(y)
plt.title('Signal')
plt.xlabel('Time (Proben)')
plt.ylabel('Amplitude')
plt.show()

spec = np.abs(librosa.stft(y, hop_length = 512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
librosa.display.specshow(spec, sr = sr, x_axis = 'time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()


n_ffft = 2048
ft = np.abs(librosa.stft(y[:n_ffft], hop_length = n_ffft+1))
plt.plot(ft)
plt.title('Spectrum')
plt.xlabel('Frequency Bin')
plt.ylabel('Amplitude')
plt.show()