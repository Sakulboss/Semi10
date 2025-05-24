# -*- coding: utf-8 -*-

import torch

from _driver_mels import trainingdata
from cnn_data_prep import data_prep
from cnn_train_net import train, save_model_structure, get_new_filename, move_working_directory
from cnn_helpers import setup_logging, load_args



def main():
    path_to_config = None # Path to the config file, if None, use the one in this directory

    # Load arguments from JSON file
    args = load_args(path_to_config)
    data_args = args['training_data']
    model_args = args['model_settings']

    # Set up logging
    logger = setup_logging()
    logger.info(f"CUDA acceleration available: {torch.cuda.is_available()}")

    data = trainingdata(data_args, logger)


    logger.info('Training will begin')

    # Check if only a specific model should be trained or if the list with models should be continued
    if args.get('train_once', False):
        loader = data_prep(data, logger, model_args)
        trained_model = train(loader, model_args, logger)
        move_working_directory()
        torch.save(trained_model[0].state_dict(), get_new_filename('pt'))

    else:
        while True:

            # Create the data loader
            loader = data_prep(data, logger, model_args)
            trained_model, acc, epoch = train(loader, model_args, logger)

            if trained_model is not None:
                save_model_structure(trained_model, acc, epoch, logger, model_args)
                continue
            else: break


if __name__ == '__main__':
    main()