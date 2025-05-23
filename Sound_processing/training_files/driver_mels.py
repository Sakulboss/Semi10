import os.path
import numpy as np
from typing import Any

from Sound_processing.training_files.datensatz import dataset
from Sound_processing.training_files.labeler import labeler
from Sound_processing.training_files.mel_specs import mel_specs
from Sound_processing.training_files.training_data import training_data


def move_working_directory():
    """
    This function changes the current working directory (cwd) to the directory 'training_files' in which also this program is located. It is necessary so that the files and directories are created and found correctly.
    Returns:
        None
    """
    working_directory = os.getcwd()
    for i in range(3):
        if os.path.basename(working_directory) != "Sound_processing":
            os.chdir('..')
            break
    os.chdir('training_files')


def create_trainingdata(settings) -> bool:
    """
    This function creates the main datasets for the CNN. If the dataset exists earlier, it is not created again.
    Args:
        settings: main settings like the type of dataset, if it should be created new, etc.
    Returns:
        bool: True if the dataset was created, False if it already exists.
    """
    #Initialize the working directory and variables
    move_working_directory()
    model = settings.get('model', 'torch')
    size = settings.get('size', 'bienen_1')

    path = os.path.join(os.getcwd(), f'training_data_{model}_{size}.npy')
    # Check if the file already exists
    if os.path.isfile(path) and not settings.get('create_new', False): return True

    # If the file does not exist, create it
    dir_list = dataset(settings.get('size', 'bees_1'), settings)
    labels = labeler(dir_list)
    mels = mel_specs(labels, settings)
    trained_data = training_data(mels, settings)

    # Save the training data
    trained_data = np.array(trained_data, dtype=object)
    os.chdir('training_files')
    np.save(f'training_data_{model}_{size}.npy', trained_data)
    return False


def load_trainingdata(model='torch', size='bienen_1') -> tuple:
    """
    This function loads the training data from the file. The file is created in the create_trainingdata function. The correct file is found by the size and model.
    Args:
        model: which model is used ('tf' or 'torch')
        size:  which size of the dataset is used ('small', 'big', 'bienen_1')

    Returns:
        contents of the file as a tuple
    """
    data = np.load(f'training_data_{model}_{size}.npy', allow_pickle=True)
    return tuple(data)


def trainingdata(settings: dict) -> tuple:
    """
    This function is the later called function. It creates the training data and loads the file.
    Args:
        settings: main settings like the type of dataset, if it should be created new, etc.
    Returns:
        contents of the training data file as a tuple
    """
    create_trainingdata(settings)
    return load_trainingdata(settings.get('model', 'torch'), settings['size'])
    ''


def main(settings):
    """
    This function was used for testing, it now serves no purpose.
    """
    data = trainingdata(settings)
    if settings['printing']: print(data)


# This dictionary was used for testing, it now serves no purpose.
args: dict[str, Any] = {
    'printing'         : True,
    'model'            : 'torch',
    'size'             : 'bienen_1', #'big', 'small', 'bienen_1'
    'create_new'       : True,
}

if __name__=='__main__':
    main(args)
