import matplotlib.pyplot as plt
from fontTools.misc.timeTools import timestampSinceEpoch
from scipy.io import wavfile as wav
from scipy.fftpack import fft,fftfreq
import numpy as np
import wave
#import sys

rate, data = wav.read(r"C:\Users\sfz-a\Documents\Soundverarbeitung\test.wav") 
#rate, data = wave.open(r"C:\Users\sfz-a\Documents\Soundverarbeitung\test.wav") 

signal = wav.read(r"C:\Users\sfz-a\Documents\Soundverarbeitung\test.wav")
n = signal.size
timestep = 0.1
freq = fftfreq(n, d=timestep)
plt.plot(data, freq)
plt.show






fft_out = np.abs(fft(data))
num = -16000
bum = 000
data = list(data)
fft_out = list(fft_out)
del data[16000]
del fft_out[16000]
#matplotlib inline
plt.plot(data[bum:num], fft_out[bum:num])
plt.show()


"""
# Open the .wav file
soundfile = wave.open("test.wav") 

# Extract Raw Audio from Wav File
signal = soundfile.readframes(-1)
signal = np.fromstring(signal, np.int16)
fs = soundfile.getframerate()

# If Stereo
if soundfile.getnchannels() == 2:
    print("Just mono files")
    sys.exit(0)

Time = np.linspace(0, len(signal) / fs, num=len(signal))
plt.figure(1)
plt.title("Signal Wave...")
plt.plot(Time, signal)
plt.show()
"""





# Extract Raw Audio from Wav File
#signal = wav_file.wav.readframes(-1)
#signal = np.frombuffer(signal, dtype=np.int16)

# Get the frame rate
#framerate = wav_file.getframerate()

# Time axis in seconds
#time = np.linspace(0, len(signal) / framerate, num=len(signal))

# Plot the waveform
#plt.figure(1)
#plt.title('Waveform')
#plt.plot(time, signal)
#plt.show()
