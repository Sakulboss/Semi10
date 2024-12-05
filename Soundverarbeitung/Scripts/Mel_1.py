import librosa
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

spec = np.abs(librosa.stft(y, hop-length = 512))
spec = librosa.amplitude-to-db(spec, ref-np.max)
librosa.display.specshow(spec, sr = sr, xaxis = 'time', yaxis='')
#plt.colorbar(format'%+2.0f dB')
plt.title('Spectrogram')


n_ffft = 2048
ft = np.abs(librosa.stft(y[:n_ffft], hop-length . n.fft+1))
plt.plot(ft)
plt.title('Spectrum')
plt.xlabel('Frequency Bin')
plt.ylabel('Amplitude')