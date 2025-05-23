import os.path
import numpy as np

from data_file_prep import dataset
from data_labeler import labeler
from data_mel_specs import mel_specs
from data_refining import training_data


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

    os.chdir('..')
    os.chdir('files')
    size = settings.get('size', 'bees_1')
    path = os.path.join(os.getcwd(), f'training_data_torch_{size}.npy')
    # Check if the file already exists
    if os.path.isfile(path) and not settings.get('create_new', False): return True

    # If the file does not exist, create it
    logger.info(f'Creating new dataset and saving it in {path}')
    logger.info(f'Downloading and sorting the files...')
    dir_list = dataset(settings.get('size', 'bienen_1'), settings, logger)
    logger.info(f'Labeling each file...')
    labels = labeler(dir_list, settings.get('training_file_extensions', 'wav'))
    logger.info(f'Creating Mel-spectograms...')
    mels = mel_specs(labels, settings, logger)
    logger.info(f'Splitting into test and training dataset and adding the right dimensions for the model...')
    trained_data = training_data(mels, settings, logger)
    logger.info(f'Saving training data...')
    # Save the training data
    #print(trained_data)
    #trained_data = np.array(trained_data, dtype=object)

    print("DEEEBUG")
    print(type(trained_data))
    if isinstance(trained_data, (tuple, list)):
        print(len(trained_data))
        for i, elem in enumerate(trained_data):
            print(f"Element {i}: type={type(elem)}, shape={getattr(elem, 'shape', 'n/a')}")
    else:
        print("trained_data ist kein Tuple oder Liste")

    #np.save(f'training_data_torch_{size}.npy', trained_data, allow_pickle=True)
    trained_data_array = np.empty(1, dtype=object)
    trained_data_array[0] = trained_data
    np.save(f'training_data_torch_{size}.npy', trained_data_array, allow_pickle=True)
    #np.save(f'training_data_torch_{size}.npy', trained_data)
    return False


def load_trainingdata(size='bees_1') -> tuple:
    """
    This function loads the training data from the file. The file is created in the create_trainingdata function. The correct file is found by the specified size.
    Args:
        size: which size of the dataset is used ('esc50', 'bees_1')
    Returns:
        contents of the file as a tuple
    """

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ALARM (in load_trainingsdata)')
    #path = f'training_data_torch_{size}.npy'
    #data = np.load(path, allow_pickle=True)
    #return tuple(data)
    data = np.load('../files/training_data_torch_bees_1.npy', allow_pickle=True)[0]
    return data


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


