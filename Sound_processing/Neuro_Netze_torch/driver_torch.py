from Sound_processing.training_files.driver_mels import trainingdata
from Sound_processing.Neuro_Netze_torch.data_prep import data_prep
from Sound_processing.Neuro_Netze_torch.train_network_torch import train, save_model_structure, get_new_filename, move_working_directory
import torch
import logging
import json

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

def main():

    # Load arguments from JSON file
    args = load_args()
    data_args = args['training_data']
    model_args = args['model_settings']

    # Set up logging
    logger = setup_logging()
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    # Create the training data if it doesn't exist
    data = trainingdata(data_args)

    # Set up the arguments for data preparation
    x = data[2].shape
    args['input_size'] = x[2] * x[3]
    args['num_classes'] = data[1]

    # Create the data loader
    loader = data_prep(data, model_args)

    # Check if only a specific model should be trained or if the list with models should be continued
    if args.get('train_once', False):
        trained_model = train(loader, model_args)
        move_working_directory()
        torch.save(trained_model[0].state_dict(), get_new_filename('pt'))
    else:
        args['model_text'] = None
        while True:
            trained_model, accuracy = train(loader, model_args)
            if trained_model is not None:
                save_model_structure(trained_model, accuracy, model_args.get("dropbox", None),)
                continue
            else: break


specifiers = {
    'printing'         : True,
    'size'             : 'bienen_1',
    'model'            : 'torch',
    'create_new'       : False,
    'learning_rate'    : 0.001,
    'batch_size'       : 128,
    'num_epochs'       : 100,
    'epochs'           : 100,
    'min_epoch'        : 5,
    'train_once'       : False,
    #'model_text'       : 'l; conv2d; (1, 16); (3, 3); 1; (1, 1);; p; avgpool; (3, 3); 1; (1, 1);; l; conv2d; (16, 48); (3, 3); 1; (1, 1);; p; avgpool; (3, 3); 1; (1, 1);; l; conv2d; (48, 48); (3, 3); 1; (1, 1);; p; maxpool; (3, 3); 1; (1, 1);; v: view;; l; linear; (307200, 10);; l; linear; (10, 10);; l; linear; (10, 2);;',
    'dropbox'          : r'C:\Users\SFZ Rechner\PycharmProjects\Semi10\Sound_processing\Neuro_Netze_torch',
}

if __name__ == '__main__':
    main()