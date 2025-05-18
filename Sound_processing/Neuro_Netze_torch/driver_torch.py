from Sound_processing.training_files.driver_mels import trainingdata
from Sound_processing.Neuro_Netze_torch.data_prep import data_prep
from Sound_processing.Neuro_Netze_torch.train_network_torch import train, save_model_structure, get_new_filename, move_working_directory
import torch
from torch import save


def main(args):
    if args['printing']: print(torch.cuda.is_available())
    data = trainingdata(args)
    x = data[2].shape
    args['input_size'] = x[2] * x[3]
    args['num_classes'] = data[1]
    loader = data_prep(data, args)

    if args.get('train_once', False):
        trained_model = train(loader, args)
        move_working_directory()
        torch.save(trained_model[0].state_dict(), get_new_filename('pt'))
    else:
        args['model_text'] = None
        while True:
            trained_model, accuracy = train(loader, args)
            if trained_model is not None:
                save_model_structure(trained_model, accuracy)
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
    'dropbox'          : r'C:\Users\sdose\Dropbox\Semi',
}

if __name__ == '__main__':
    main(specifiers)