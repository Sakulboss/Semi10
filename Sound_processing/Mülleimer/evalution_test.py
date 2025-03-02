from Sound_processing.Neuo_Netze_torch.datensatz_torch import directory, dataset
from Sound_processing.Neuo_Netze_torch.labeler_torch import labeler
from Sound_processing.Neuo_Netze_torch.training_data_torch import training_data
#from Sound_processing.Neuro_Netze_torch.model_processing_torch import model_training_torch
#from Sound_processing.Neuro_Netze_torch.model_evaluation_torch import model_evaluation_torch
from Sound_processing.Neuo_Netze_torch.mel_specs_torch import mel_specs

import sys

file_ = sys.stdout
printing = True

def main(settings):
    directory()

    dir_list = dataset(settings) #passt
    print(1)
    labels = labeler(dir_list, settings)
    print(2)
    mels = mel_specs(labels, settings)
    print(mels)
    print(3)
    trained_data = training_data(mels, settings)
    #print(*trained_data)
    print(4)
    #trained = model_training_torch(trained_data, settings)
    #model_evaluation_torch(trained_data, trained[0], settings)
    return trained_data


args = {
    'printing'         : True,
    'big'              : False,
    'plot_history'     : True,
    'confusion_matrix' : True,
}

if __name__=='__main__':
    main(args)