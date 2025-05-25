import os.path, logging
import numpy as np

from data_file_prep import dataset
from data_labeler import labeler
from data_mel_specs import mel_specs
from data_refining import refine_data


def setup_logging(args:dict) -> logging.Logger:
    handlers = []
    if args.get('log_to_file', False):   logging.FileHandler(args.get('log_file', 'training.log'))
    if args.get('log_to_console', True): handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=args.get('level', 2),
        format=args.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=handlers
    )
    return logging.getLogger(__name__)

def create_trainingdata(settings: dict, logging_args:dict) -> bool:
    """
    This function creates the main datasets for the CNN. If the dataset exists earlier, it is not created again.
    Args:
        logging_args: dict with arguments for logging
        settings:     dict main settings like the type of dataset, if it should be created new, etc.
    Returns:
        bool: True if the dataset was created, False if it already exists.
    """
    logger = setup_logging(logging_args)

    #Initialize the working directory and variables
    os.chdir('..')
    os.chdir('files')
    size = settings.get('size', 'bees_1')
    path = os.path.join(os.getcwd(), f'training_data_torch_{size}.npy')
    # Check if the file already exists
    if os.path.isfile(path) and not settings.get('create_new', False):
        logger.info(f'Dataset {size} already exists, skipping creation.')
        return True

    #Chain of creating the training data
    logger.info(f'Creating new dataset and saving it in {path}.')
    logger.info(f'Downloading and sorting the files...')
    dir_list = dataset(settings.get('size', 'bienen_1'), settings, logging_args)
    logger.info(f'Labeling each file...')
    labels = labeler(dir_list, settings.get('training_file_extensions', 'wav'))
    logger.info(f'Creating Mel-spectograms...')
    mels = mel_specs(labels, settings, logging_args)
    logger.info(f'Splitting into test and training dataset and adding the right dimensions for the model...')
    trained_data = refine_data(mels)
    logger.info(f'Saving training data...')

    # Save the training data
    trained_data_array = np.empty(1, dtype=object)
    trained_data_array[0] = trained_data

    # Save the array as numpy file; needs pickle, because object is a non-primitive (tuple)
    np.save(f"training_data_torch_{size}.npy", trained_data_array, allow_pickle=True)
    return False


def load_trainingdata(size:str ='bees_1') -> tuple:
    """
    This function loads the training data from the file. The file is created in the create_trainingdata function. The correct file is found by the specified size.
    Args:
        size: str which one of the dataset is used ('esc50', 'bees_1')
    Returns:
        contents of the file as a tuple
    """

    data = np.load(f"../files/training_data_torch_{size}.npy", allow_pickle=True)[0]
    return data


def trainingdata(settings:dict, logging_args:dict) -> tuple:
    """
    This function is the later called function. It creates the training data and loads the file.
    Args:
        logging_args: dict with arguments for logging
        settings:     dict with main settings like the type of dataset, if it should be created new, etc.
    Returns:
        contents of the training data file as a tuple
    """

    create_trainingdata(settings, logging_args)
    return load_trainingdata(settings.get('size', 'bees_1'))


