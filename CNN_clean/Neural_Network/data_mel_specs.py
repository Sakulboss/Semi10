import logging
import numpy as np
import librosa.feature as mf
import librosa
from tqdm import tqdm


def setup_logging(args: dict) -> logging.Logger:
    handlers = []
    if args.get('log_to_file', False):   logging.FileHandler(args.get('log_file', 'training.log'))
    if args.get('log_to_console', True): handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=args.get('level', 2),
        format=args.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=handlers
    )
    return logging.getLogger(__name__)

def mel_spec_file(fn_wav_name, logging_args, n_fft=1024, hop_length=441, fss = 48000, n_mels=64, stereo:bool=True) :
    """
    Compute mel cepstrogram from audio file with librosa.feature.melspectogram()
    Args:
        fn_wav_name:  str   Audio file name
        logging_args: dict  with arguments for logging
        n_fft:        int   FFT size
        hop_length:   int   Hop size in samples
        fss:          float Sample rate in Hz
        n_mels:       int   Number of mel-bands
        stereo:       bool  If False, convert to mono
    Returns:
        np.ndarray: Mel cepstrogram
    """
    logger = setup_logging(logging_args)

    try:
        x_new, fss = librosa.load(fn_wav_name, sr=fss, mono=not stereo)
    except Exception as e:
        logger.error(e)
        return None

    if stereo:
        x_new = np.append(x_new[0], x_new[1])

    # normalize to the audio file to a maximum absolute value of 1
    if np.max(np.abs(x_new)) > 0:
        x_new = x_new / np.max(np.abs(x_new))

    x_new = mf.melspectrogram(y=x_new,
                                sr=fss,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                n_mels=n_mels,
                                fmin=0.0,
                                fmax=fss / 2,
                                power=1.0,
                                htk=True,
                                norm=None
                              )

    # apply dB normalization
    x_new = librosa.amplitude_to_db(x_new)

    return x_new


def mel_specs(labels: tuple, setting, logging_args) -> tuple:
    """
    Create mel cepstrogram from audio files and splits them into segments to generate more training data.
    Args:
        labels:       tuple of data from previous step
        setting:      dict  with main settings like the type of dataset and injection of other labeled data.
        logging_args: dict  with arguments for logging
    Returns:
        segment_file_mod_id: list of file ids of the segments
        segment_list:        list of mel cepstrogram segments
        segment_class_id:    list of class ids of the segments
        data[2]:             list of class names
        data[5]:             int  of classes
    """
    logger = setup_logging(logging_args)

    # Initialize variables
    size = setting.get('size', 'bienen_1')
    data = setting.get('classified_samples', labels)
    fn_wav_list = setting.get('fn_wav_list', data[0])
    class_id = setting.get('class_id', data[1])
    segments_per_spectrogram = int(setting.get('segments_per_spectrogram', 10))
    segment_length_frames = int(setting.get('segment_length_frames', 100))

    all_mel_specs = []
    segment_list = []
    segment_file_id = []
    segment_class_id = []
    error_files = []

    # Create mel cepstrogram
    for count in tqdm(range(len(fn_wav_list)), desc='Mel-Cepstogramm'):
        mel_spec = mel_spec_file(fn_wav_list[count], logging_args, stereo=(size == 'bees_1'))
        if mel_spec is None or count != 0 and mel_spec.shape != all_mel_specs[-1].shape:
            error_files.append(fn_wav_list[count])
        else:
            all_mel_specs.append(mel_spec)
    # check for faulty files (can very rarely happen, when recording in flac)
    if not error_files == []:
        error_files = [(str(i) + "\n") for i in error_files]
        logger.error(f'Wrong file size, please remove the file(s): {error_files}')

    # combine the mel specs into one np.array
    all_mel_specs = np.stack(all_mel_specs, axis=0)
    max_segment_start_offset = all_mel_specs.shape[-1] - segment_length_frames

    # Create segments from the mel cepstrogram with random start points
    for i in range(len(all_mel_specs)):
        # create n segments from each cepstrogram
        for s in range(segments_per_spectrogram):
            # Use as many parts of the mel cepstrogram as possible, if more are needed, select a random start point
            if (s+1) * segment_length_frames <= all_mel_specs.shape[-1]:
                segment_list.append(all_mel_specs[i, :, s*segment_length_frames:(s+1)*segment_length_frames])
            else:
                segment_start_frames = int(np.random.rand(1).item() * max_segment_start_offset)
                segment_list.append(all_mel_specs[i, :, segment_start_frames:segment_start_frames + segment_length_frames])
            segment_file_id.append(i)
            segment_class_id.append(class_id[i])

    # conversion from the list of cepstrogram segments into a np.array for faster processing
    segment_list = np.array(segment_list)
    segment_file_id = np.array(segment_file_id)
    segment_file_mod_id = np.mod(segment_file_id, 5)
    segment_class_id = np.array(segment_class_id)

    return segment_file_mod_id, segment_list, segment_class_id, data[2], data[5]


