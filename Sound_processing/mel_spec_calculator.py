import librosa
import numpy as np
import os
import wget
import zipfile
import shutil


def compute_mel_spec_for_audio_file(fn_wav_name, n_fft=1024, hop_length=441, fss = 22050., n_mels=64, mono = True, channel = 0):
    """ Compute mel spectrogram
    Args:
        fn_wav_name (str): Audio file name
        n_fft (int): FFT size
        hop_length (int): Hop size in samples
        fss (float): Sample rate in Hz
        n_mels (int): Number of mel bands
        mono (bool): If True, convert to mono
    """
    # load audio samples
    x_new, fss = librosa.load(fn_wav_name, sr=fss, mono=mono)
    if not mono:
        x_new = x_new[channel]
    # normalize to the audio file to a maximum absolute value of 1
    if np.max(np.abs(x_new)) > 0:
        x_new = x_new / np.max(np.abs(x_new))
    # mel-spectrogram
    x_new = librosa.feature.melspectrogram(y=x_new,
                                       sr=fss,
                                       n_fft=n_fft,
                                       hop_length=hop_length,
                                       n_mels=n_mels,
                                       fmin=0.0,
                                       fmax=fss / 2,
                                       power=1.0,
                                       htk=True,
                                       norm=None)
    # apply dB normalization
    x_new = librosa.amplitude_to_db(x_new)

    return x_new

def get_new_filename(file_extension: str) -> str:
    count = len([counter for counter in os.listdir('modelle') if counter.endswith(file_extension)]) + 1
    return f'full_model_{count}.{file_extension}'

if __name__ == '__main__':
    print(get_new_filename('keras'))
    print(compute_mel_spec_for_audio_file('bird.wav').shape)