from Sound_processing.training_files.driver_mels import trainingdata
from Sound_processing.Neuro_Netze_torch.data_prep import data_prep
from Sound_processing.Neuro_Netze_torch.train_network_torch import train, save_model_structure
import torch

print(torch.cuda.is_available())
def main(args):
    data = trainingdata(args)
    x = data[2].shape
    args['input_size'] = x[2] * x[3]
    args['num_classes'] = data[1]
    loader = data_prep(data, args)
    while True:
        trained_model = train(loader, args)
        if trained_model is not None:
            save_model_structure(trained_model, args)
            continue
        else: break


specifiers = {
    'printing'         : True,
    'size'             : 'small',
    'plot_history'     : False,
    'confusion_matrix' : False,
    'model'            : 'torch',
    'learning_rate'    : 0.001,
    'batch_size'       : 64,
    'num_epochs'       : 100,
    'epochs'           : 100,
}

if __name__ == '__main__':
    main(specifiers)