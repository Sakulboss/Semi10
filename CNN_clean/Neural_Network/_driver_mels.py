import os.path
import numpy as np

from data_file_prep import dataset
from data_labeler import labeler
from data_mel_specs import mel_specs
from data_refining import training_data


def move_working_directory():
    """
    This function changes the current working directory (cwd) to the directory 'training_files' in which also this program is located. It is necessary so that the files and directories are created and found correctly.
    Returns:
        None
    """
    working_directory = os.getcwd()
    for i in range(3):
        if os.path.basename(working_directory) != "Sound_processing":
            os.chdir('../..')
            break
    os.chdir('training_files')


def create_trainingdata(settings, logger) -> bool:
    """
    This function creates the main datasets for the CNN. If the dataset exists earlier, it is not created again.
    Args:
        logger: logger for logging
        settings: main settings like the type of dataset, if it should be created new, etc.
    Returns:
        bool: True if the dataset was created, False if it already exists.
    """
    #Initialize the working directory and variables
    #move_working_directory()
    model = settings.get('model', 'torch')
    size = settings.get('size', 'bienen_1')

    path = os.path.join(os.getcwd(), f'training_data_{model}_{size}.npy')
    # Check if the file already exists
    if os.path.isfile(path) and not settings.get('create_new', False): return True

    # If the file does not exist, create it
    dir_list = dataset(settings.get('size', 'bienen_1'), settings)
    labels = labeler(dir_list, settings.get('training_file_extensions', 'wav'))
    mels = mel_specs(labels, settings, logger)
    trained_data = training_data(mels, settings)

    # Save the training data
    trained_data = np.array(trained_data, dtype=object)
    os.chdir('..')
    os.chdir('training_files')
    np.save(f'training_data_{model}_{size}.npy', trained_data)
    return False


def load_trainingdata(size='bees_1') -> tuple:
    """
    This function loads the training data from the file. The file is created in the create_trainingdata function. The correct file is found by the specified size.
    Args:
        size: which size of the dataset is used ('esc50', 'bees_1')
    Returns:
        contents of the file as a tuple
    """
    data = np.load(f'training_data_torch_{size}.npy', allow_pickle=True)
    return tuple(data)


def trainingdata(settings: dict, logger) -> tuple:
    """
    This function is the later called function. It creates the training data and loads the file.
    Args:
        logger: logger for logging
        settings: main settings like the type of dataset, if it should be created new, etc.
    Returns:
        contents of the training data file as a tuple
    """
    create_trainingdata(settings, logger)
    return load_trainingdata(settings.get('size', 'bees_1'))


