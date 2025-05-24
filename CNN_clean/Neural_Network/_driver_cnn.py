# -*- coding: utf-8 -*-

import torch
import logging

from tqdm.contrib.logging import logging_redirect_tqdm

from _driver_mels import trainingdata
from cnn_data_prep import data_prep
from cnn_train_net import train, save_model_structure, get_new_filename, move_working_directory
from cnn_helpers import load_args


def setup_logging(args):
    handlers = []
    if args.get('log_to_file', False):   logging.FileHandler(args.get('log_file', 'training.log'))
    if args.get('log_to_console', True): handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=args.get('level', 2),
        format=args.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=handlers
    )
    return logging.getLogger(__name__)



def main():


    path_to_config = None # Path to the config file, if None, use the one in this directory

    # Load arguments from JSON file
    args = load_args(path_to_config)
    data_args = args['training_data']
    model_args = args['model_settings']
    logging_args = args['logger_settings']

    logger = setup_logging(logging_args)

    # Set up logging
    logging.getLogger('numba.core.byteflow').setLevel(logging.WARNING)
    logging.getLogger('numba.core.interpreter').setLevel(logging.WARNING)

    logger.info(f"CUDA acceleration available: {torch.cuda.is_available()}")

    data = trainingdata(data_args, logging_args)


    logger.info('Training will begin')

    # Check if only a specific model should be trained or if the list with models should be continued
    if args.get('train_once', False):
        loader = data_prep(data, logging_args, model_args)
        trained_model = train(loader, model_args, logging_args)
        move_working_directory()
        torch.save(trained_model[0].state_dict(), get_new_filename('pt'))

    else:
        while True:

            # Create the data loader
            loader = data_prep(data, logging_args, model_args)
            trained_model, acc, epoch = train(loader, logging_args, model_args)

            if trained_model is not None:
                save_model_structure(trained_model, acc, epoch, logging_args, model_args)
                continue
            else: break


if __name__ == '__main__':
    main()