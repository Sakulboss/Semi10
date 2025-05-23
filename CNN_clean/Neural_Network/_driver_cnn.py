# -*- coding: utf-8 -*-

import torch
from threading import Thread

from _driver_mels import trainingdata
from cnn_data_prep import data_prep
from cnn_train_net import train, save_model_structure, get_new_filename, move_working_directory
from cnn_helpers import setup_logging, get_uuid, load_args

waiting = False


def main():
    global waiting
    path_to_config = None # Path to the config file, if None, use the one in this directory

    # Load arguments from JSON file
    args = load_args(path_to_config)
    data_args = args['training_data']
    model_args = args['model_settings']

    # Set up logging
    logger = setup_logging()
    logger.info(f"CUDA acceleration available: {torch.cuda.is_available()}")

    # Create the training data if it doesn't exist
    data = trainingdata(data_args, logger)

    # Set up the arguments for data preparation
    x = data[2].shape
    args['input_size'] = x[2] * x[3]
    args['num_classes'] = data[1]

    # Create the data loader
    loader = data_prep(data, model_args)
    logger.info('Training will begin')

    # Check if only a specific model should be trained or if the list with models should be continued
    if args.get('train_once', False):
        trained_model = train(loader, model_args, logger)
        move_working_directory()
        torch.save(trained_model[0].state_dict(), get_new_filename('pt'))

    else:
        while True:
            #waiting = True
            #thread_animation = Thread(target=animate)
            #thread_animation.start()

            trained_model = train(loader, model_args, logger)
            #waiting = False

            if trained_model is not None:
                save_model_structure(trained_model, logger, model_args)
                continue
            else: break


if __name__ == '__main__':
    main()