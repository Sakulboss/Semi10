import numpy as np
import wget
import os
import matplotlib
import librosa
#%matplotlib inline #-nur für das Notebook
import matplotlib.pyplot as pl
import platform
import IPython.display as ipd

# check current platform
print(platform.platform())

# use if/else conditions
if "Windows" in platform.platform():
    dir_audio = 'C:/audio_files'
else:
    dir_audio = '/my_server/audio_files'

if not os.path.isfile('piano.wav') or not os.path.isfile('bird.wav'):
    for fn in ('piano.wav', 'bird.wav'):
        wget.download('https://github.com/machinelistening/machinelistening.github.io/blob/master/{}?raw=true'.format(fn),
                      out=fn, bar=None)
else:
    print('Files already exist!')

# 1) define path to the directory that contains the audio files (WAV format)
# TIP: under Windows, it is also recommended to use '/', e.g. 'C:/my_audio_files'
dir_wav = ''  # here, we use the same directory as the notebook is in

# this could also look like
# dir_wav = 'c:/audio_files'

# 2) create absolute path of audio file (directory + filename)
# os.path.join takes care of the correct delimiter signs
# - Linux / MacOSx: "/"
# - Windows: "\\"

fn_wav = os.path.join(dir_wav, 'bird.wav')  # original filename: 416529__inspectorj__bird-whistling-single-robin-a_2s
print(fn_wav)

# (1) use the sample rate of the file, load stereo if needed
x, fs = librosa.load(fn_wav)

print("Sample vector shape:", x.shape)  # 1D numpy array, mono
print("Sample rate [Hz]", fs)
print(f"Audio duration (seconds): {len(x)/fs}")

# (2) you could also enforce another sample rate
fs_fix = 44100
x, fs = librosa.load(fn_wav, sr=fs_fix)  # in this case, the signal is upsampled to a higher sample rate

print(x.shape)  # ! increase of sampling rate (upsampling) -> more samples!
print(fs) # ! fix sample rate was used

# (3) if you have a stereo file, you can enforce one channel audio (mono)
# x, fs = librosa.load(fn_wav, mono=True)

ipd.display(ipd.Audio(data=x, rate=fs))

sample_end = int(fs*1.5)
x_first_1_5_s = x[:sample_end]
print(f"Our segment has a duration of {len(x_first_1_5_s)/fs} seconds.")

pl.figure(figsize=(10,2))
pl.plot(x)
pl.show()

number_of_samples = len(x)
print("Number of samples:", number_of_samples)

seconds_per_sample = 1/fs
print("Duration [seconds] of one sample", seconds_per_sample)  # on audio sample corresponds to ~22.7 ms

# let's create a numpy array with the physical time of each audio sample
frames_in_seconds = np.arange(number_of_samples)*seconds_per_sample


pl.figure(figsize=(10,2))
pl.plot(frames_in_seconds, x)
pl.xlabel('Time [s]')
pl.ylabel('Amplitude')
pl.show()

n_fft = 2048
hop_length = 1024
X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)  # using the default values: n_fft=2048,
print("Shape of STFT:", X.shape)  # we get n_fft//2 - 1 bins, the reason is that the STFT has a
                                  # symmetric structure and we can discard several entries
print("Data type of STFT:", X.dtype)  # ! the STFT is complex and has a magnitude and a phase

# We'll focus on the magnitude of the STFT
S = np.abs(X)
print("Shape of the magnitude spectrogram:", S.shape)
print("Data type of the magnitude spectrogram:", S.dtype)  # ok

# let's plot it
pl.figure(figsize=(6,4))
pl.imshow(S, aspect="auto")
pl.colorbar()
pl.show()

# Issue 1.) use "origin" parameter for imshow()
pl.figure(figsize=(6,4))
pl.imshow(S, aspect="auto", origin="lower")
pl.colorbar()
pl.show()

# Issue 2.) apply logarithmic compression to the magnitude values -> this converts the linear magnitudes to decibels (dB)
S_dB = librosa.amplitude_to_db(S, ref=np.max)

fig = pl.figure(figsize=(6,4))
pl.imshow(S_dB, aspect="auto", origin="lower")
pl.colorbar(format='%+2.0f dB')
pl.show()

# Issue 3.) define maximum frequency, and maximum time value
f_max = fs/2  # Nyquist frequency
t_max = number_of_samples / fs
print(f_max)
print(t_max)

pl.figure(figsize=(6,4))
# use "extent" parameter to define actual range of values along x / y acis
pl.imshow(S_dB, aspect="auto", origin="lower", extent=[0, t_max, 0, f_max])
pl.xlabel('Time [s]')
pl.ylabel('Frequency [Hz]')
pl.colorbar(format='%+2.0f dB')
pl.show()

fig, ax = pl.subplots()
# note the keyword "mel", which indicates that a mel frequency axis is used
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='hz', sr=fs, ax=ax, cmap='viridis')
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='STFT Magnitude Spectrogram (specshow)')
pl.show()

M = librosa.feature.melspectrogram(y=x, n_fft=2048, hop_length=1024, n_mels=128)
print("Shape of Mel spectrogram:", M.shape)  # frequency x time: we get n_mels frequency bands

# apply dB compression as before
M_dB = librosa.amplitude_to_db(M, ref=np.max)

fig, ax = pl.subplots()
# note the keyword "mel", which indicates that a mel frequency axis is used
img = librosa.display.specshow(M_dB, x_axis='time', y_axis='mel', sr=fs, ax=ax, cmap='viridis')
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
pl.show()

fn_wav = os.path.join(dir_wav, 'piano.wav')  # original filename: 196765__xserra__piano-phrase.wav
x, fs = librosa.load(fn_wav)
ipd.display(ipd.Audio(data=x, rate=fs))

n_octaves = 5  # let's capture 5 octaves starting from C1
bins_per_octave = 12  # let's choose a frequency resolution of 100 cent (= one frequency bin per semitone)
C = np.abs(librosa.cqt(x, sr=fs, n_bins=n_octaves*bins_per_octave , bins_per_octave=bins_per_octave))
print("Shape of CQT:", C.shape)  # logically, we get 60 frequency bins

# dB magnitude scaling
C = librosa.amplitude_to_db(C)

# we can use again the visualization tool provided by librosa
fig, ax = pl.subplots()
img = librosa.display.specshow(C, sr=fs, x_axis='time', y_axis='cqt_note', ax=ax)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
pl.show()

bins_per_octave = 50
C = np.abs(librosa.cqt(x, sr=fs, n_bins=n_octaves*bins_per_octave , bins_per_octave=bins_per_octave))
print("Shape of CQT:", C.shape)
C = librosa.amplitude_to_db(C)
fig, ax = pl.subplots()
img = librosa.display.specshow(C, sr=fs, x_axis='time', y_axis='cqt_note', ax=ax)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
pl.show()