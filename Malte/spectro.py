import numpy as np
from scipy.signal import spectrogram
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import librosa
import os
import random
import soundfile as sf


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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_two_spectrograms(cmap, spec1, spec2, title1='Mel-Spectrogram 1', title2='Mel-Spectrogram 2', fig_width=14, fig_height=5, save_path=True):
    label_left = 5
    pad = 20
    vmin = min(spec1.min(), spec2.min())
    vmax = max(spec1.max(), spec2.max())

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    img1 = ax1.imshow(spec1, origin='lower', aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_title(title1)
    ax1.tick_params(axis='y', pad=pad)
    ax1.set_xlabel('Time (frames)')
    ax1.set_ylabel('Mel Frequency Bands', labelpad=label_left)

    a=0.09

    color='yellow'
    line_color='red'
    '''
    red_lines = [9, 14, 20, 30, 39]
    for y in red_lines:
        ax1.axhline(y=y, color=line_color, linestyle='--', linewidth=1)

    ax1.axhline(y=50, color=line_color, linestyle='dashdot', linewidth=1)
    ax1.axhline(y=60, color=line_color, linestyle='dashdot', linewidth=1)

    highlight_areas = [(9,14),(20,30),(39,45),(56,59)]
    for start, end in highlight_areas:
        ax1.axhspan(start, end, facecolor=color, alpha=a)

    arrow_position = [12, 22, 45, 55]
    for y in arrow_position:
        ax1.annotate(
            '',
            xy=(-0.005, y),  # Pfeilspitze: ganz links an der Achse (x=0 in Achsenkoordinaten)
            xycoords=('axes fraction', 'data'),  # x in Achsenkoordinaten, y in Datenkoordinaten
            xytext=(-0.05, y),  # Pfeilanfang: 10% links außerhalb der Achse (x=-0.1 in Achsenkoordinaten)
            textcoords=('axes fraction', 'data'),
            arrowprops=dict(
                arrowstyle='-|>',
                color='red',
                lw=3
            )
        )
    '''

    ax2 = fig.add_subplot(gs[1])
    img2 = ax2.imshow(spec2, origin='lower', aspect='auto', interpolation='none', vmin=vmin, vmax=vmax, cmap=cmap)
    ax2.set_title(title2)
    ax2.set_xlabel('Time (frames)')
    ax2.tick_params(axis='y', pad=pad)
    ax2.set_ylabel('Mel Frequency Bands', labelpad=label_left)
    '''
    red_lines = [9, 14, 20, 30, 42]
    for y in red_lines:
        ax2.axhline(y=y, color=line_color, linestyle='--', linewidth=1)
    ax2.axhline(y=49, color=line_color, linestyle='dashdot', linewidth=1)
    ax2.axhline(y=60, color=line_color, linestyle='dashdot', linewidth=1)
    
    highlight_areas = [(9,14),(20,30),(42,49)]
    for start, end in highlight_areas:
        ax2.axhspan(start, end, facecolor=color, alpha=a)
    
    arrow_position = [12,22,33,45,55]
    for y in arrow_position:
        ax2.annotate(
            '',
            xy=(-0.005, y),  # Pfeilspitze: ganz links an der Achse (x=0 in Achsenkoordinaten)
            xycoords=('axes fraction', 'data'),  # x in Achsenkoordinaten, y in Datenkoordinaten
            xytext=(-0.05, y),  # Pfeilanfang: 10% links außerhalb der Achse (x=-0.1 in Achsenkoordinaten)
            textcoords=('axes fraction', 'data'),
            arrowprops=dict(
                arrowstyle='-|>',
                color='red',
                lw=3
            )
        )
    '''

    cbar_ax = fig.add_subplot(gs[2])
    cbar = fig.colorbar(img2, cax=cbar_ax, format='%+2.0f dB')
    cbar.set_label('Amplitude (dB)')

    plt.tight_layout()
    plt.subplots_adjust(left=0.06, right=0.95)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Funktion zur Anzeige der Colormaps
def plot_colormaps():
    colormap_categories = {
        'Perceptually Uniform Sequential': [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ],
        'Sequential': [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        ],
        'Sequential (2)': [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper'
        ],
        'Diverging': [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
        ],
        'Cyclic': ['twilight', 'twilight_shifted', 'hsv'],
        'Qualitative': [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c'
        ],
        'Miscellaneous': [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'
        ]
    }

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack([gradient] * 2)

    fig_height = sum(len(maps) for maps in colormap_categories.values()) * 0.22
    fig, axs = plt.subplots(nrows=sum(len(maps) for maps in colormap_categories.values()),
                            figsize=(10, fig_height))
    fig.subplots_adjust(top=1, bottom=0, left=0.2, right=0.99)

    i = 0
    for category, colormaps in colormap_categories.items():
        for cmap in colormaps:
            axs[i].imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap))
            axs[i].text(-0.01, 0.5, cmap, va='center', ha='right', fontsize=9,
                        transform=axs[i].transAxes)
            axs[i].set_axis_off()
            i += 1

    plt.show()

def main():
    #path = r"F:\Aufnahmen\Zuhause\3-4_Mai\output_2025-05-02-18-32-59_13_17.wav" #9
    #path = r"F:\Aufnahmen\Zuhause\2_Mai\output_2025-05-02-02-33-52_19_17.wav"
    #path = r"F:\Aufnahmen\Zuhause\3-4_Mai\output_2025-05-04-08-00-41_2107_17.wav" #8
    #path = r"F:\Aufnahmen\Zuhause\4-5_Mai\output_2025-05-04-15-19-58_87_17.wav" #7
    #path = r"F:\Aufnahmen\Zuhause\4-5_Mai\output_2025-05-05-10-01-39_1165_17.wav" #6
    #path = r"F:\Aufnahmen\Zuhause\6-7_Mai\output_2025-05-06-07-11-12_367_17.wav" #5
    #path = r"F:\Aufnahmen\Zuhause\7-8_Mai\output_2025-05-07-11-36-22_2_17.wav" #4
    #path = r"F:\Aufnahmen\Zuhause\7-8_Mai\output_2025-05-08-09-10-54_1213_17.wav" #3
    path = r"F:\Aufnahmen\Zuhause\8-9_Mai\output_2025-05-09-03-15-29_433_17.wav" #2


    v = 2
    data, sr = sf.read(path)
    left_channel = data[:, 0]
    right_channel = data[:, 1]
    left_wav = f"left_temp_{1}.wav"
    right_wav = f"right_temp_{2}.wav"
    sf.write(left_wav, left_channel, sr)
    sf.write(right_wav, right_channel, sr)
    left_spec = mel_spec_file(left_wav)
    right_spec = mel_spec_file(right_wav)
    liste=['Accent', 'BuPu','gnuplot2','Greys','inferno','Pastel1','Pastel2','terrain']
    for i in liste:
        plot_two_spectrograms(
            cmap=i,
            spec1=left_spec,
            spec2=right_spec,
            title1='Keine Schwarmstimmung',
            title2=f'Schwarmstimmung, {v} Tage vor Schwarmereignis',
            fig_width=18,
            fig_height=5,
            save_path=f'C:/Users/SFZ Rechner/Desktop/{v} Tage/schwarmstimmung_{i}_{v}_tage_vor_event.png'
        )

    os.remove(left_wav)
    os.remove(right_wav)

if __name__ == '__main__':
    main()

