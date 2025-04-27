from Sound_processing.training_files.driver_mels import trainingdata
from Sound_processing.Neuro_Netze_torch.data_prep import data_prep
from Sound_processing.Neuro_Netze_torch.train_network_torch import train


def main(args):
    data   = trainingdata(args)
    x = data[2].shape
    args['input_size'] = x[2] * x[3]
    args['num_classes'] = data[1]
    loader = data_prep(data, args)
    train(loader, args)


specifiers = {
    'printing'         : True,
    'size'             : 'small',
    'plot_history'     : False,
    'confusion_matrix' : False,
    'model'            : 'torch',
    'learning_rate'    : 0.001,
    'batch_size'       : 64,
    'num_epochs'       : 10,
    'epochs'           : 10,
}

if __name__ == '__main__':
    main(specifiers)