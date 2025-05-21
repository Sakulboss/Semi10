import numpy as np
import librosa.feature as mf
import librosa

def mel_spec_file(fn_wav_name, n_fft=1024, hop_length=441, fss = 22050., n_mels=64, stereo:bool=True):
    """
    Compute mel spectrogram from audio file with librosa.feature.melspectogram()
    Args:
        fn_wav_name (str): Audio file name
        n_fft (int): FFT size
        hop_length (int): Hop size in samples
        fss (float): Sample rate in Hz
        n_mels (int): Number of mel-bands
        stereo (bool): If False, convert to mono
    Returns:
        x_new (ndarray): Mel spectrogram
    """

    try:
        x_new, fss = librosa.load(fn_wav_name, sr=fss, mono=not stereo)
    except Exception as e:
        print(e)
        return None

    if stereo:
        x_new = np.append(x_new[0], x_new[1])

    #normalize to the audio file to a maximum absolute value of 1
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


def mel_specs(labels, setting):
    """
    Create mel spectrograms from audio files and splits them into segments to generate more training data.
    Args:
        labels: data from previous step
        setting: main settings like the type of dataset and injection of other labeled data.
    Returns:
        segment_file_mod_id: file ids of the segments
        segment_list: list of mel spectrogram segments
        segment_class_id: class ids of the segments
        data[2]: class names
        data[5]: number of classes
    """

    # Initialize variables
    size = setting.get('size', 'small')
    data = setting.get('classified_samples', labels)
    fn_wav_list = setting.get('fn_wav_list', data[0])
    class_id = setting.get('class_id', data[1])
    all_mel_specs = []
    segment_list = []
    segment_file_id = []
    segment_class_id = []
    error_files = []

    # Create mel spectrograms
    for count, fn_wav in enumerate(fn_wav_list):
        mel_spec = mel_spec_file(fn_wav_list[count], stereo=(size == 'bienen_1'))
        if mel_spec is None or count != 0 and mel_spec.shape != all_mel_specs[-1].shape:
            error_files.append(fn_wav_list[count])
        else:
            all_mel_specs.append(mel_spec)
    if not error_files:
        error_files = [(str(i) + "\n") for i in error_files]
        print(f'Wrong file size, please remove the file(s): {error_files}')

    all_mel_specs = np.stack(all_mel_specs, axis=0)

    #Überprüfen der Form jedes aufgeteilten Arrays
    print(all_mel_specs)
    n_spectrograms = all_mel_specs.shape[0]
    spec_length_frames = all_mel_specs.shape[2]

    n_segments_per_spectrogram = 10
    segment_length_frames = 100

    max_segment_start_offset = spec_length_frames - segment_length_frames

    # Create segments from the mel spectrograms with random start points
    for i in range(n_spectrograms):
        # create ... segments from each spectrogram
        for s in range(n_segments_per_spectrogram):
            segment_start_frames = int(np.random.rand(1).item() * max_segment_start_offset)
            segment_list.append(all_mel_specs[i, :, segment_start_frames:segment_start_frames + segment_length_frames])
            segment_file_id.append(i)
            segment_class_id.append(class_id[i])


    # conversion from the list of spectrogram segments into a tensor (3D array)
    segment_list = np.array(segment_list)

    segment_file_id = np.array(segment_file_id)
    segment_file_mod_id = np.mod(segment_file_id, 5)
    segment_class_id = np.array(segment_class_id)


    return segment_file_mod_id, segment_list, segment_class_id, data[2], data[5]


"""
    for count, fn_wav in enumerate(fn_wav_list):
        mel_spec = mel_spec_file(fn_wav_list[count], stereo=(True if size == 'bienen_1' else False))
        all_mel_specs.append(mel_spec)

    np.stack(all_mel_specs, axis=0)

    #Überprüfen der Form jedes aufgeteilten Arrays
    print(all_mel_specs)

[array([[-33.791866  ,  22.673313  ,  46.208134  , ...,  -5.124857  ,
         -4.2888856 ,  -9.720901  ],
       [-33.791866  ,  22.482834  ,  34.55688   , ...,  -2.6883914 ,
         -0.38356927,  -3.9091184 ],
       [-33.791866  ,  17.50881   ,  22.943785  , ...,  -4.7874737 ,
          1.4943385 ,  -9.132776  ],
       ...,
       [-33.791866  , -33.791866  , -19.334375  , ...,  -9.002867  ,
         -6.7380486 , -10.176064  ],
       [-33.791866  , -33.791866  , -20.55895   , ...,  -7.74972   ,
         -8.871878  , -11.370798  ],
       [-33.791866  , -33.791866  , -25.03178   , ..., -14.853021  ,
        -14.045716  , -14.814632  ]], dtype=float32), array([[ 39.936672  ,  46.11798   ,  42.456852  , ...,  -0.52852446,
        -15.067913  , -19.264408  ],
       [ 33.55336   ,  26.988716  ,  18.72043   , ...,   0.31303194,
         -5.825963  , -17.693495  ],
       [ 22.061148  ,  10.892081  ,   6.091602  , ...,   5.2239504 ,
         -3.2982874 ,  -9.63443   ],
       ...,
       [ -4.1767526 ,  -4.8684173 ,  -5.0880985 , ...,  -7.634582  ,
         -7.6387005 , -14.453521  ],
       [ -5.3656006 ,  -6.674224  ,  -5.5583534 , ...,  -8.22873   ,
         -9.205669  , -10.662287  ],
       [-12.203167  , -12.096238  ,  -9.366321  , ..., -12.010111  ,
        -15.118695  , -18.118704  ]], dtype=float32), array([[ 40.076164  ,  46.18133   ,  42.56168   , ...,  -8.626376  ,
         -3.5325594 ,  -9.598931  ],
       [ 33.40502   ,  26.633215  ,  19.290369  , ...,  -6.479719  ,
         -8.222304  ,  -5.327016  ],
       [ 22.141891  ,   9.767334  ,   3.7966478 , ...,  -9.696088  ,
          0.5626113 ,   0.11929905],
       ...,
       [ -3.6210723 ,  -7.7227907 ,  -8.5599165 , ...,  -8.755479  ,
         -8.505269  , -12.043179  ],
       [ -4.2818527 ,  -8.623446  ,  -9.28945   , ..., -10.043999  ,
         -9.272462  , -13.953742  ],
       [ -9.238412  , -13.814703  , -13.447737  , ..., -14.838237  ,
        -14.360429  , -15.643446  ]], dtype=float32), array([[ 40.111942 ,  46.098    ,  42.397743 , ...,  -7.35066  ,
         -1.9997467,  -8.4619665],
       [ 33.348114 ,  26.845036 ,  18.661602 , ...,  -5.8271637,
         -4.7210984,  -4.064646 ],
       [ 21.075487 ,  12.354902 ,   3.8516128, ...,  -1.7595005,
         -5.5188723,  -5.811246 ],
       ...,
       [ -6.0483737,  -5.197413 ,  -6.7663097, ...,  -8.535603 ,
         -9.447899 , -11.687588 ],
       [ -4.553736 ,  -5.179403 ,  -5.934435 , ...,  -8.077659 ,
         -8.286363 , -12.790081 ],
       [-12.063362 , -11.825891 , -14.814981 , ..., -12.687683 ,
        -15.113248 , -14.262391 ]], dtype=float32), array([[ 39.85856   ,  46.17029   ,  42.492287  , ..., -11.509266  ,
         -6.736464  ,  -7.007468  ],
       [ 33.35405   ,  26.85939   ,  18.484234  , ...,  -8.533136  ,
         -4.6573734 ,  -6.5492654 ],
       [ 21.741726  ,  11.841016  ,   5.992913  , ...,  -0.46773273,
         -7.751704  ,  -5.253761  ],
       ...,
       [ -2.4128718 ,  -5.5263243 ,  -4.6237316 , ...,  -8.45573   ,
         -9.068207  ,  -6.3729944 ],
       [ -3.1553311 ,  -7.453285  ,  -3.8235989 , ...,  -6.1944733 ,
         -8.684708  ,  -8.913414  ],
       [-10.073647  , -12.23098   ,  -9.59779   , ..., -12.724907  ,
        -13.390387  , -10.153395  ]], dtype=float32), array([[ 39.74737   ,  46.015663  ,  42.39922   , ...,  -6.2971153 ,
         -1.0256426 ,  -7.531933  ],
       [ 33.15297   ,  26.553381  ,  18.89859   , ...,  -3.0854516 ,
         -0.12742296,  -7.090191  ],
       [ 21.465069  ,  10.921831  ,   5.608504  , ...,   1.3858824 ,
         -0.41401786,  -7.1204214 ],
       ...,
       [ -4.4671755 ,  -6.9592576 ,  -7.4274817 , ...,  -9.733246  ,
         -7.1185894 , -11.881359  ],
       [ -4.9094963 ,  -9.128576  ,  -7.4707127 , ...,  -8.165171  ,
        -10.0903425 , -12.925358  ],
       [-11.343059  , -14.911215  , -13.397039  , ..., -12.41651   ,
        -13.098197  , -17.084291  ]], dtype=float32), array([[ 40.123516 ,  46.218304 ,  42.484318 , ...,  -7.583571 ,
         -6.9834256,  -6.9454365],
       [ 33.53174  ,  27.074587 ,  18.856714 , ...,  -2.1482933,
         -6.1302233,  -5.2197113],
       [ 22.001427 ,  10.302686 ,   8.826902 , ...,   4.386985 ,
          0.6898452, -11.486694 ],
       ...,
       [ -4.6545825,  -9.733554 ,  -9.971806 , ...,  -6.760609 ,
         -7.7771816,  -4.8727145],
       [ -5.360862 ,  -8.931295 ,  -8.270718 , ...,  -8.131479 ,
         -9.41264  ,  -7.2618527],
       [-12.762217 , -16.017803 , -11.881687 , ..., -13.278275 ,
        -15.067598 ,  -8.741084 ]], dtype=float32), array([[ 40.007114 ,  46.091682 ,  42.52335  , ...,  -6.8715277,
         -3.9804971,   0.6525957],
       [ 33.270412 ,  26.772898 ,  19.221493 , ...,  -3.483329 ,
         -2.059178 ,   2.2893174],
       [ 20.893213 ,  12.332506 ,   5.5593085, ...,   3.7352128,
          2.2110965,   3.6062036],
       ...,
       [ -6.5781794,  -6.027856 ,  -6.8939524, ...,  -5.6386547,
         -8.804329 , -11.613413 ],
       [ -7.1379967,  -7.9254174,  -5.4907713, ...,  -5.23559  ,
         -5.341998 , -11.9463625],
       [-13.43745  , -14.153236 , -13.395487 , ..., -10.918051 ,
        -11.054297 , -16.23171  ]], dtype=float32), array([[ 39.808887 ,  46.10802  ,  42.539616 , ..., -11.224901 ,
         -2.0504951, -15.011357 ],
       [ 33.119934 ,  26.76505  ,  18.436512 , ..., -12.072476 ,
         -2.001275 , -13.744249 ],
       [ 21.686117 ,   8.942799 ,   2.7357464, ...,   1.79598  ,
         -5.242514 ,  -3.670144 ],
       ...,
       [ -6.784977 ,  -5.7743893,  -6.095669 , ...,  -7.1248217,
         -9.818558 ,  -8.640634 ],
       [ -5.02358  ,  -7.487005 ,  -7.229953 , ...,  -8.910333 ,
         -8.48404  ,  -8.830844 ],
       [-11.45279  , -12.5974455, -13.934746 , ..., -14.494494 ,
        -14.57499  , -14.060212 ]], dtype=float32), array([[ 39.940254  ,  46.22864   ,  42.67128   , ..., -14.71663   ,
         -5.503245  , -14.6099615 ],
       [ 33.395695  ,  27.17894   ,  19.32968   , ..., -11.741552  ,
         -0.76623607, -11.574935  ],
       [ 22.225512  ,  11.693183  ,   3.188332  , ...,  -1.4950633 ,
         -2.6875792 , -11.073042  ],
       ...,
       [ -5.491992  ,  -3.4238863 ,  -2.0522795 , ...,  -7.637924  ,
         -5.596505  ,  -8.602193  ],
       [ -5.4107385 ,  -5.7145576 ,  -2.7980673 , ...,  -7.111833  ,
         -5.5727367 ,  -8.23309   ],
       [ -9.3393545 ,  -9.515443  ,  -7.4747515 , ..., -10.926718  ,
        -10.680172  ,  -7.573656  ]], dtype=float32)]


"""