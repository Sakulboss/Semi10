import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import struct
import sys
import wave
from pathfinder import soundpath

#fft - eigen -------------------------------------------------
def DFT(x):
    """
    Function to calculate the
    discrete Fourier Transform
    of a 1D real-valued signal x
    """
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(e, x)
    return X


def dft_eigen():
    # anwendung ------------------------------------------
    sr, x = wavfile.read(soundpath("air_raid.wav"))
    X = DFT(x)

    # calculate the frequency
    N = len(X)
    n = np.arange(N)
    T = N / sr
    freq = n / T

    plt.figure(figsize=(8, 6))
    plt.stem(freq, abs(X), 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('DFT Amplitude |X(freq)|')
    plt.show()

    n_oneside = N // 2
    # get the one side frequency
    f_oneside = freq[:n_oneside]

    # normalize the amplitude
    X_oneside = X[:n_oneside] / n_oneside

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.stem(f_oneside, abs(X_oneside), 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('DFT Amplitude |X(freq)|')

    plt.subplot(122)
    plt.stem(f_oneside, abs(X_oneside), 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.xlim(0, 10)
    plt.tight_layout()
    plt.show()
    """
    # sampling rate
    sr = 100
    # sampling interval
    ts = 1.0/sr
    t = np.arange(0,1,ts)

    freq = 1.
    x = 3*np.sin(2*np.pi*freq*t)

    freq = 4
    x += np.sin(2*np.pi*freq*t)

    freq = 7
    x += 0.5* np.sin(2*np.pi*freq*t)

    plt.figure(figsize = (8, 6))
    plt.plot(t, x, 'r')
    plt.ylabel('Amplitude')

    plt.show()"""

#wavesin.py -----------------------------------------------------------
def wavesin():
    # Read the wave file
    sample_rate, data = wavfile.read(soundpath("test.wav"))

    # Create a time array
    time = np.linspace(0, len(data) / sample_rate, num=len(data))
    nu = 1000
    print(len(time), len(data))

    dft_eigen(data, sample_rate)

    # Plot the wave file
    plt.figure(figsize=(10, 4))
    plt.plot(time[:nu], data[:nu])
    plt.title('Wave File as Sine Wave')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.show()


#123.py ---------------------------------------------------------------
def run123():
    spf = wave.open(soundpath("test.wav"), "r")

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, np.int16)
    fs = spf.getframerate()

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)


    Time = np.linspace(0, len(signal) / fs, num=len(signal))

    plt.figure(1)
    plt.title("Signal Wave...")
    plt.plot(Time, signal)
    plt.show()


#FastFourierTransformation.py ------------------------------------------------------------------------
def fastfuriertransformation():
    rate, data = wavfile.read(soundpath("test.wav"))
    # rate, data = wave.open(r"C:\Users\sfz-a\Documents\Soundverarbeitung\test.wav")

    signal = wavfile.read(r"/Soundverarbeitung/test.wav")
    n = signal.size
    timestep = 0.1
    freq = fftfreq(n, d=timestep)
    plt.plot(data, freq)
    plt.show()


#jksrgh.py -------------------------------------------------------------
def maxelement(yfNew, freq):
    delindex = yfNew.index(max(yfNew))
    del yfNew[delindex]
    del freq[delindex]

def jksrgh():
    # Loading data (could be from any time-series data source)
    data = np.random.randn(1024) # Example data
    sample_rate, data = wavfile.read(soundpath("test.wav"))
    # Fourier transform and frequency analysis
    N = len(data)
    T = N / sample_rate
    yf = fft(data)
    freq = list(fftfreq(N, T))

    yfNew = list(np.abs(yf))

    for i in range(100):
        maxelement(yfNew, freq)

    # Plotting
    plt.plot(freq, yfNew)
    plt.title('Frequency Spectrum')
    plt.ylabel('Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.show()


# fsdg.py ----------------------------------------------------------------------------------
def fsdg():
    # Lade die WAV-Datei
    sample_rate, data = wavfile.read(soundpath("test.wav"))

    # Berechne die Fourier-Transformation
    N = len(data)
    T = 1.0 / sample_rate
    yf = fft(data)
    xf = fftfreq(N, T)[:N // 2]

    # Plotten des Frequenzspektrums
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.grid()
    plt.show()


# teesz.py -----------------------------------------------------------------------------------------
def teesz():
    wavefile = wave.open(soundpath("test.wav"), 'r')

    length = wavefile.getnframes()

    numbers = 1
    liste123 = []
    for i in range(0, length):
        # wavedata = wavefile.readframes(1)
        # data = struct.unpack("<h", wavedata)

        wavedata = wavefile.readframes(numbers)
        data = struct.unpack(f"<{numbers}h", wavedata)

        liste123.append(int(data[0]))

    # This snippet reads 1 frame. To read more than one frame (e.g., 13), use
    plt.title("Signal Wave...")
    plt.plot(liste123)
    plt.show()


# test.py ------------------------------------------------------------------------
def test():
    spf = wave.open(soundpath("test.wav"), 'r')

    # Extract Raw Audio from Wav File
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, np.int16)
    fs = spf.getframerate()

    # If Stereo
    if spf.getnchannels() == 2:
        print("Just mono files")
        sys.exit(0)

    Time = np.linspace(0, len(signal) / fs, num=len(signal))

    plt.figure(1)
    plt.title("Signal Wave...")
    plt.plot(Time, signal)
    plt.show()

