import numpy as np
import matplotlib.pyplot as pl
import librosa
import sys

printing: bool = False
file_ = sys.stdout

def printer(*text, sep=' ', end='\n', file=None):
    global printing
    global file_
    file = file or file_
    printer(*text, sep=sep, end=end, file=file)


def mel_spec_file(fn_wav_name, n_fft=1024, hop_length=441, fss = 22050., n_mels=64):
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


def mel_specs(labels, **setting):
    """
    Args:
        labels:
        **setting:
    """
    global printing, file_
    printing = setting.get('print', False)
    file_ = setting.get('file', sys.stdout)

    data = setting.get('classified_samples', labels)
    fn_wav_list = setting.get('fn_wav_list', data[0])
    class_id = setting.get('class_id', data[1])
    all_mel_specs = []
    printing = setting.get('printing', False)

    for count, fn_wav in enumerate(fn_wav_list):
        all_mel_specs.append(mel_spec_file(fn_wav_list[count]))

    printer("We have {} spectrograms of shape: {}".format(len(all_mel_specs), all_mel_specs[0].shape))

    all_mel_specs = np.array(all_mel_specs)
    printer(f"Shape of our data tensor:         {all_mel_specs.shape}")

    segment_list = []
    segment_file_id = []
    segment_class_id = []
    segment_spec_id = []

    n_spectrograms = all_mel_specs.shape[0]

    n_segments_per_spectrogram = 10
    segment_length_frames = 100

    spec_length_frames = all_mel_specs.shape[2]
    max_segment_start_offset = spec_length_frames - segment_length_frames

    # iterate over all spectrograms
    for i in range(n_spectrograms):

        # create [n_segments_per_spectrogram] segments
        for s in range(n_segments_per_spectrogram):
            # random segment start frame
            segment_start_frames = int(np.random.rand(1).item() * max_segment_start_offset)

            segment_list.append(all_mel_specs[i, :, segment_start_frames:segment_start_frames + segment_length_frames])

            segment_file_id.append(i)
            segment_class_id.append(class_id[i])
            segment_spec_id.append(s)

    # finally, let's convert our list of spectrogram segments again into a 3D tensor
    segment_list = np.array(segment_list)

    segment_file_id = np.array(segment_file_id)
    segment_file_mod_id = np.mod(segment_file_id, 5)

    segment_class_id = np.array(segment_class_id)
    segment_spec_id = np.array(segment_spec_id)

    printer(f"New data tensor shape:            {segment_list.shape}")

    if setting.get('file_ID_diagram', False):
        pl.figure(figsize=(12, 4))
        pl.plot(segment_file_id, 'b-', label='segment file ID')
        pl.plot(segment_file_mod_id, 'b--', label='segment file ID (per spectrogram)')
        pl.plot(segment_class_id, label='segment class ID')
        pl.plot(segment_spec_id, label='segment ID')
        pl.legend()
        pl.xlabel('Segment')
        pl.show()

    if setting.get('plot_spectrogram', False):
        pl.figure(figsize=(2.5, 2))
        pl.imshow(all_mel_specs[0, :, :], origin="lower", aspect="auto", interpolation="None")
        pl.xticks([], [])
        pl.yticks([], [])
        pl.title('Original spectrogram')
        pl.tight_layout()
        pl.show()

    if setting.get('plot_segments', False):
        pl.figure(figsize=(15, 5))
        ny = 2
        nx = int(n_segments_per_spectrogram // ny)
        for s in range(n_segments_per_spectrogram):
            pl.subplot(ny, nx, s + 1)
            pl.imshow(segment_list[s, :, :], origin="lower", aspect="auto", interpolation="None")
            if s == 0:
                pl.title('Extracted segments')
            pl.xticks([], [])
            pl.yticks([], [])
        pl.tight_layout()
        pl.show()
    return segment_file_mod_id, segment_list, segment_class_id, data[2], data[5]

if __name__ == '__main__':
    pass