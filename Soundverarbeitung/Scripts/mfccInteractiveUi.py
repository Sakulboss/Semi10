import librosa.display
import ipywidgets as widgets
from IPython.display import display
from IPython.display import Audio
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import numpy as np
from pathfinder import soundpath


# File uploader widget
uploader = widgets.FileUpload(accept='.wav', multiple=False)

# Load and compute MFCCs for a given audio file
def compute_mfcc(file):
    y, sr = librosa.load(soundpath(file), sr=None)
    pre_emphasis = 0.97
    y_preemphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
    signal_length = len(y_preemphasized)
    frame_size = 0.025  # 25 ms
    frame_stride = 0.01  # 10 ms
    frame_length, frame_step = frame_size * sr, frame_stride * sr
    NFFT = 512
    nfilt = 40
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    num_ceps = 12

    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(y_preemphasized, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]

    plt.figure(figsize=(14, 5))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.subplot(2, 1, 2)
    plt.imshow(mfcc.T, cmap='hot', aspect='auto')
    plt.title('MFCC')
    plt.xlabel('Frame Index')
    plt.ylabel('Cepstral Coefficient Index')
    plt.tight_layout()
    plt.show()

# Handle file upload and compute MFCCs
def on_upload_change(change):
    file = list(uploader.value.values())[0]
    compute_mfcc(file['content'])

uploader.observe(on_upload_change, names='value')

# Button to compute MFCCs for the sample audio file
sample_button = widgets.Button(description='Use Sample Audio')
def on_sample_button_click(b):
    compute_mfcc(sample_audio_path)

sample_button.on_click(on_sample_button_click)

# Display widgets
display(uploader, sample_button)