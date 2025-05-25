# -*- coding: utf-8 -*-
import torch, logging, os, json

from _driver_mels import trainingdata
from cnn_data_prep import data_prep
from cnn_train_net import train, save_model_structure, get_new_filename, move_working_directory


def setup_logging(args: dict) -> logging.Logger:
    """
    This function sets up the logger. Each file has its own, but it is configured the same in every file.
    Args:
        args: dict The arguments for the logger, such as the level and handlers (like console logging)
    Returns:
        The logger object
    """
    handlers = []
    if args.get('log_to_file', False):   handlers.append(logging.FileHandler(args.get('log_file', 'training.log')))
    if args.get('log_to_console', True): handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=args.get('level', 2),
        format=args.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_args(path:str =None) -> dict:
    """
    This function loads the configuration file.
    Args:
        path: str Path to the config.json file. If None, it will use the one in the current working directory.
    Returns:
        configuration as dictionary
    """
    if path is None:
        path = os.path.join(os.getcwd(), 'config.json')

    with open(path, 'r') as file:
        args = json.load(file)
    return args


def main() -> None:
    """
    Main function to run the CNN training process.
    Returns:
        None
    """


    # Load arguments from JSON file, leave path_to_config as None to use the default config.json in the current directory
    path_to_config = None
    args = load_args(path_to_config)
    data_args = args['training_data']
    model_args = args['model_settings']
    logging_args = args['logger_settings']

    # Set up logging
    logger = setup_logging(logging_args)

    # These loggers are set to warning because in debug they produce too much output
    logging.getLogger('numba.core.byteflow').setLevel(logging.WARNING)
    logging.getLogger('numba.core.interpreter').setLevel(logging.WARNING)

    # CUDA is the programm to compute on NVIDEA GPUs, with workarounds also for AMD GPUs
    logger.info(f"CUDA acceleration available: {torch.cuda.is_available()}")
    data = trainingdata(data_args, logging_args)
    logger.info('Training will begin..')

    # Check for continous training
    if model_args.get('train_once', False):
        # Create and train the model
        loader = data_prep(data, logging_args, model_args)
        trained_model = train(loader, logging_args, model_args)

        # save the model structure
        move_working_directory('models')
        torch.save(trained_model[0].state_dict(), get_new_filename('pt'))
        logger.info('Training ended and the model was saved.')

    else:
        while True:
            # Create and train the model
            loader = data_prep(data, logging_args, model_args)
            trained_model, acc, epoch = train(loader, logging_args, model_args)

            # Save the model structure (possibly to the server) if the model recieved a line, otherwise break
            if trained_model is not None:
                save_model_structure(trained_model, acc, epoch, logging_args, model_args)
                continue
            else: break

        logger.info('Training ended, because no more models are available for training.')

# Run the main function
if __name__ == '__main__':
    main()