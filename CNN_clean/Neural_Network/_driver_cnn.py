# -*- coding: utf-8 -*-

import torch
import logging
import json
import time
from threading import Thread

from _driver_mels import trainingdata
from cnn_data_prep import data_prep
from cnn_train_net import train, save_model_structure, get_new_filename, move_working_directory

waiting = False


def setup_logging():
    # Konfiguriere das Logging-Format und Level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # logging.FileHandler('training.log'),
            logging.StreamHandler()  # Ausgabe auch in der Konsole
        ]
    )
    return logging.getLogger(__name__)


def load_args() -> dict:
    """
    This function loads the arguments from a file. The file is created in the create_trainingdata function. The correct file is found by the size and model.
    Returns:
        contents of the file as a dictionary
    """
    with open('config.json', 'r') as file:
        args = json.load(file)
    return args


def animate():
    global waiting
    animation = "|/-\\"
    idx = 0
    while waiting:
        print('Netz wird trainiert ', animation[idx % len(animation)], end="\r")
        idx += 1
        time.sleep(0.1)


def main():
    global waiting
    # Load arguments from JSON file
    args = load_args()#
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
        model_args['model_text'] = None
        while True:
            #waiting = True
            #thread_animation = Thread(target=animate)
            #thread_animation.start()

            trained_model, accuracy = train(loader, model_args, logger)
            #waiting = False

            if trained_model is not None:
                save_model_structure(trained_model, accuracy, model_args.get("dropbox", None))
                continue
            else: break


if __name__ == '__main__':
    main()