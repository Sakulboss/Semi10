import os.path
import numpy as np
from typing import Any

from Sound_processing.training_files.datensatz import dataset
from Sound_processing.training_files.labeler import labeler
from Sound_processing.training_files.mel_specs import mel_specs
from Sound_processing.training_files.training_data import training_data

def create_trainingdata(settings) -> bool:
    model = settings.get('model', 'torch')
    size = settings.get('size', 'small')

    if os.path.isfile(f'training_data_{model}_{size}'): return True

    dir_list = dataset(settings)
    labels = labeler(dir_list, settings)
    mels = mel_specs(labels, settings)
    trained_data = training_data(mels, settings, model=model)

    trained_data = np.array(trained_data, dtype=object)
    os.chdir('training_files')
    np.save(f'training_data_{model}_{size}.npy', trained_data)
    return False

def load_trainingdata():
    ladung = np.load('training_data_tf_small.npy', allow_pickle=True)
    return tuple(ladung)

def trainingdata(settings: dict) -> tuple:
    """

    Returns:
        object: 
    """
    create_trainingdata(settings)
    return load_trainingdata()

def main(settings):
    trainingdata(settings)
    
args: dict[str, Any] = {
    'printing' : True,
    'model' : 'torch',
    'size' : 'small'
}

if __name__=='__main__':
    main(args)