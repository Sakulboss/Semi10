from sklearn.metrics import confusion_matrix

from datensatz_torch import directory, dataset
from labeler_torch import labeler
from training_data_torch import training_data
from model_processing_torch import model_training_torch
from mel_specs_torch import mel_specs
from model_evaluation_torch import model_evaluation

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
    trained = model_training_torch(trained_data, settings)
    model_evaluation(trained_data, trained[0], settings)

printing = False
file_ = sys.stdout

args = {
    'printing' : True,
    'big'      : False,
    'confusion_matrix' : True,
}

if __name__=='__main__':
    main(args)