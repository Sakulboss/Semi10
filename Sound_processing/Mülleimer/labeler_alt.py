import numpy as np
import os
import glob
import sys


def labeler(data: list, settings):
    """
    Args:
        data:
        settings:
    """

    big = settings.get('big', False)
    sub_directories = settings.get('sub_directories', data)
    n_sub = len(sub_directories)
    printing = settings.get('printing', False)
    # let's collect the files in each subdirectory
    # the folder name is the class name
    fn_wav_list = []
    class_label = []
    file_num_in_class = []

    for i in range(n_sub):
        current_class_label = os.path.basename(sub_directories[i])
        current_fn_wav_list = sorted(glob.glob(os.path.join(sub_directories[i], '*.wav')))
        for k, fn_wav in enumerate(current_fn_wav_list):
            fn_wav_list.append(fn_wav)
            class_label.append(current_class_label)
            file_num_in_class.append(k)

    n_files = len(class_label)

    # this vector includes a "counter" for each file within its class, we use it later ...
    file_num_in_class = np.array(file_num_in_class)

    unique_classes = sorted(list(set(class_label)))
    if printing: print("All unique class labels (sorted alphabetically): ", unique_classes)
    class_id = np.array([unique_classes.index(_) for _ in class_label])

    return fn_wav_list, class_id, unique_classes, n_files, file_num_in_class, n_sub