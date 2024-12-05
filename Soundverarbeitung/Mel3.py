import librosa
import matplotlib.pyplot as plt
import numpy as np
#import scipy.fftpack
import spect

def main():
    abstand_spectogramm = 500

    # Signal ausgeben, sr= Samplerate
    y, sr = librosa.load('bienensummen.wav')

    # Signal FFT
    n_fft = 2048
    ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft + 1))

    # Spectrogramm ohne Mel's
    spec = np.abs(librosa.stft(y, hop_length=abstand_spectogramm))
    spec = librosa.amplitude_to_db(spec, ref=np.max)

    # Mel_spectogram_1

    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    mel_spect = librosa.power_to_db(spect, ref = np.max)

    # Mel_Spectogramm_2

    # n_fft = Anzahl der Werte in jeder STFT (standard: 2048) hop_length = Number of samples between successive STFT
    # frames (default: n_fft // 2) (wie viele abschnitte wollen wir haben? n_mels = Anzahl der Mel-Banden die
    # generiert werden (standart 128) fmin & fmax = minimum und maximum der frequenzen in den Mel-Koeffizienten (standard durch librosa.filters.mel())
    fmin = 0
    fmax = 1500
    #Mel Spectrogramm wird extrahiert
    mel_spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, fmax = fmax, fmin = fmin)

    # ref = Weißt daraufhin Amplitude zu nutzen, wenn Umwandlung in Decibel
    #Lineares Spectrogramm wird zu Decibel (logarithmus) umgewandelt
    # cmap = colormap die genutzt wird (cmap='viridis')
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref = np.max)


    #Diagramme
    """
    #Signalamplituden
    plt.plot(y)
    plt.title('Signal')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.show() """

    """
    #FFT erster Frame
    plt.plot(ft)
    plt.title('Spectrum')
    plt.xlabel('Frequency Bin')
    plt.ylabel('Amplitude')
    plt.show()"""

    """
    #Spektogramm
    librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.show()"""


    #Mel Spectogram 1?
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    # Mel Spectrogramm plotten
    plt.figure(figsize=(8, 8))
    librosa.display.specshow(mel_spectrogram_db, x_axis = 'time', y_axis = 'mel', sr = sr, fmax = fmax) #'time'
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()

if __name__ == "__main__":
    main()