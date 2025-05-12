import numpy as np
import matplotlib.pyplot as plt
import librosa
import sys

def mel_spec_file(fn_wav_name, n_fft=1024, hop_length=441, fss=2000., n_mels=64):
    """ Compute mel spectrogram
    Args:
        fn_wav_name (str): Audio file name
        n_fft (int): FFT size
        hop_length (int): Hop size in samples
        fss (float): Sample rate in Hz
        n_mels (int): Number of mel bands
    """
    # Load audio samples
    x_new, fss = librosa.load(fn_wav_name, sr=fss, mono=True)

    # Normalize to the audio file to a maximum absolute value of 1
    if np.max(np.abs(x_new)) > 0:
        x_new = x_new / np.max(np.abs(x_new))

    # Compute mel-spectrogram
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

    # Apply dB normalization
    x_new = librosa.amplitude_to_db(x_new)

    return x_new

def plot_spectrogram(spectrogram, title='Mel-Spectrogram'):
    """ Plot the mel spectrogram
    Args:
        spectrogram (np.ndarray): Mel spectrogram to plot
        title (str): Title of the plot
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, origin='lower', aspect='auto', interpolation='None')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time (frames)')
    plt.ylabel('Mel Frequency Bands')
    plt.tight_layout()
    plt.show()

def main():
    input_wav = 'mono_left.wav'
    print(f"Processing file: {input_wav}")

    # Compute mel spectrogram
    mel_spectrogram = mel_spec_file(input_wav)

    # Plot the mel spectrogram
    plot_spectrogram(mel_spectrogram)

if __name__ == '__main__':
    main()
