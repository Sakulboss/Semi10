import librosa
import numpy as np
import os
import wget
import zipfile
import shutil

def download_dataset(printing=False):
    if not os.path.isfile('animal_sounds.zip'):
        if printing: print('\nPlease wait a couple of seconds ...')
        try:
            wget.download(
                'https://github.com/machinelistening/machinelistening.github.io/blob/master/animal_sounds.zip?raw=true',
                out='animal_sounds.zip', bar=None)
            if printing: print('animal_sounds.zip downloaded successfully ...')
        except Exception as e:
            if printing: print(f"Error downloading file: {str(e)}")

    else:
        if printing: print('\nFiles already exist!', '\n')

    if not os.path.isdir('animal_sounds'):
        if printing: print("\nLet's unzip the file ... ")
        assert os.path.isfile('animal_sounds.zip')
        with zipfile.ZipFile('animal_sounds.zip', 'r') as f:
            # unzip all files into current folder
            f.extractall('.')
        assert os.path.isdir('animal_sounds')
        if printing: print("All done :)", '\n')

def big_dataset(printing=False):
    source_folder = 'viele_sounds'
    target_base_folder = 'viele_sounds_geordnet'
    eintraege = []
    directories = []
    with open('esc50.csv', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                eintraege.append(line.split(','))
    if not os.path.isdir(target_base_folder):
        os.mkdir(target_base_folder)
    for i in range(len(eintraege)):
        target_folder = os.path.join(target_base_folder, eintraege[i][3])
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)
        target_file = os.path.join(target_folder, eintraege[i][0])
        if not os.path.isfile(target_file):
            source_file = os.path.join(source_folder, eintraege[i][0])
            shutil.copy(source_file, target_file)
        directories.append(eintraege[i][3] + '/' + eintraege[i][0])
    if printing: print(len(directories))


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

def get_new_filename(file_extension: str) -> str:
    count = len([counter for counter in os.listdir('.\\modelle') if counter.endswith(file_extension)]) + 1
    return f'full_model_{count}.{file_extension}'

if __name__ == '__main__':
    print(get_new_filename('keras'))