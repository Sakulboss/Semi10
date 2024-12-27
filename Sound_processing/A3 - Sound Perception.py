import numpy as np
import wget
import os
import matplotlib
import librosa
#%matplotlib inline
import matplotlib.pyplot as pl
import platform
import IPython.display as ipd
import zipfile

if not os.path.isfile('piano.wav') or not os.path.isfile('c_maj_inv_2_sounds.wav'):
    for fn in ('piano.wav', 'c_maj_inv_2_sounds.wav'):
        wget.download('https://github.com/machinelistening/machinelistening.github.io/blob/master/{}?raw=true'.format(fn),
                      out=fn, bar=None)
else:
    print('Files already exist!')

fn_wav = 'piano.wav'
x, fs = librosa.load(fn_wav)
ipd.display(ipd.Audio(data=x, rate=fs))

n_octaves = 5  # let's capture 5 octaves starting from C1
bins_per_octave = 12  # let's choose a frequency resolution of 100 cent (= one frequency bin per semitone)
C = np.abs(librosa.cqt(x, sr=fs, n_bins=n_octaves*bins_per_octave , bins_per_octave=bins_per_octave))
C = librosa.amplitude_to_db(C)  # dB magnitude scaling
fig, ax = pl.subplots()
img = librosa.display.specshow(C, sr=fs, x_axis='time', y_axis='cqt_note', ax=ax)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
pl.show()

hop_length=128
cens = librosa.feature.chroma_cens(y=x, sr=fs, hop_length=hop_length, fmin=None, tuning=None, n_chroma=12, n_octaves=7, bins_per_octave=36)
t_max = len(x) / fs

pl.figure()
pl.imshow(cens, aspect="auto", interpolation="None", origin="lower", extent=[0, t_max, 0, cens.shape[0]])
# let's create an interpretable pitch axis
pl.yticks([0.5, 2.5, 4.5, 5.5, 7.5, 9.5, 11.5], ['C', 'D', 'E', 'F', 'G', 'A', 'B'])
pl.xlabel('Time (seconds)')
pl.ylabel('Pitch Class')
pl.title('CENS')
pl.tight_layout()
pl.show()

image_url = "https://github.com/machinelistening/machinelistening.github.io/blob/master/c_maj_inv_2_sounds.png?raw=true"
ipd.display(ipd.Image(url=image_url))

fn_wav = 'c_maj_inv_2_sounds.wav'
x, fs = librosa.load(fn_wav)
ipd.display(ipd.Audio(data=x, rate=fs))

hop_length=128
cens = librosa.feature.chroma_cens(y=x, sr=fs, hop_length=hop_length, fmin=None, tuning=None, n_chroma=12, n_octaves=7, bins_per_octave=36)
t_max = len(x) / fs

pl.figure()
pl.imshow(cens, aspect="auto", interpolation="None", origin="lower", extent=[0, t_max, 0, cens.shape[0]])
# let's create an interpretable pitch axis
pl.yticks([0.5, 2.5, 4.5, 5.5, 7.5, 9.5, 11.5], ['C', 'D', 'E', 'F', 'G', 'A', 'B'])
pl.xlabel('Time (seconds)')
pl.ylabel('Pitch Class')
pl.title('CENS')
pl.tight_layout()
pl.show()

n_octaves = 7  # let's capture 7 octaves starting from C1
bins_per_octave = 12  # let's choose a frequency resolution of 100 cent (= one frequency bin per semitone)
C = np.abs(librosa.cqt(x, sr=fs, n_bins=n_octaves*bins_per_octave , bins_per_octave=bins_per_octave))
C = librosa.amplitude_to_db(C)  # dB magnitude scaling
fig, ax = pl.subplots()
img = librosa.display.specshow(C, sr=fs, x_axis='time', y_axis='cqt_note', ax=ax)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
pl.show()

# let's compute and visualize 13 MFCCs
n_mfcc = 20
n_fft = 2048
hop_length = 1024
mf = librosa.feature.mfcc(y=x, sr=fs, S=None, n_mfcc=n_mfcc, dct_type=2, norm='ortho', lifter=0, n_fft=n_fft, hop_length=hop_length)
pl.figure()
pl.imshow(mf, aspect="auto", origin="lower", extent=[0, t_max, 0, mf.shape[0]])
pl.xlabel('Time [s]')
pl.ylabel('MFCC')
pl.show()

