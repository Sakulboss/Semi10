import librosa
import numpy as np
import os
import wget
import zipfile

def download_dataset():
    if not os.path.isfile('animal_sounds.zip'):
        print('\nPlease wait a couple of seconds ...')
        try:
            wget.download(
                'https://github.com/machinelistening/machinelistening.github.io/blob/master/animal_sounds.zip?raw=true',
                out='animal_sounds.zip', bar=None)
            print('animal_sounds.zip downloaded successfully ...')
        except Exception as e:
            print(f"Error downloading file: {str(e)}")

    else:
        print('\nFiles already exist!', '\n')

    if not os.path.isdir('animal_sounds'):
        print("\nLet's unzip the file ... ")
        assert os.path.isfile('animal_sounds.zip')
        with zipfile.ZipFile('animal_sounds.zip', 'r') as f:
            # unzip all files into current folder
            f.extractall('.')
        assert os.path.isdir('animal_sounds')
        print("All done :)", '\n')

def compute_mel_spec_for_audio_file(fn_wav_name, n_fft=1024, hop_length=441, fss = 22050., n_mels=64):
    """ Compute mel spectrogram
    Args:
        fn_wav_name (str): Audio file name
        n_fft (int): FFT size
        hop_length (int): Hop size in samples
        fss (float): Sample rate in Hz
        n_mels (int): Number of mel bands
    """
    # load audio samples
    x_new, fss = librosa.load(fn_wav_name, sr=fss, mono=True)

    # normalize to the audio file to an maximum absolute value of 1
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
