import numpy as np
import os
import glob
import sys


def labeler(data: list):
    """
    This function labels the data. It takes a list of paths to subdirectories and creates a list of the file names with their labels.
    Args:
        data: list of paths to subdirectories
    Returns:
        fn_wav_list: list of file names
        class_id: list of class ids - easier for computation
        unique_classes: list of class labels - used for human readability
        n_files: number of all used files
        file_num_in_class: list of file numbers in class - used for human readability
        n_sub: number of subdirectories

    """

    n_sub = len(data)

    fn_wav_list = []
    class_label = []
    file_num_in_class = []

    for i in range(n_sub):
        current_class_label = os.path.basename(data[i])
        current_fn_wav_list = sorted(glob.glob(os.path.join(data[i], '*.wav')))
        for k, fn_wav in enumerate(current_fn_wav_list):
            fn_wav_list.append(fn_wav)
            class_label.append(current_class_label)
            file_num_in_class.append(k)

    n_files = len(class_label)

    file_num_in_class = np.array(file_num_in_class)

    unique_classes = sorted(list(set(class_label)))
    class_id = np.array([unique_classes.index(_) for _ in class_label])

    return fn_wav_list, class_id, unique_classes, n_files, file_num_in_class, n_sub
