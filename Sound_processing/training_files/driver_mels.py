import os.path
import numpy as np
from typing import Any

from Sound_processing.training_files.datensatz import dataset
from Sound_processing.training_files.labeler import labeler
from Sound_processing.training_files.mel_specs import mel_specs
from Sound_processing.training_files.training_data import training_data

def move_working_directory():
    working_directory = os.getcwd()
    for i in range(3):
        if os.path.basename(working_directory) != "Sound_processing":
            os.chdir('..')
            break
    os.chdir('training_files')

def create_trainingdata(settings) -> bool:
    model = settings.get('model', 'torch')
    size = settings.get('size', 'small')

    move_working_directory()
    path = os.path.join(os.getcwd(), f'training_data_{model}_{size}.npy')

    if os.path.isfile(path): print('Mels exist!'); return True

    dir_list = dataset(settings)
    labels = labeler(dir_list, settings)
    mels = mel_specs(labels, settings)
    trained_data = training_data(mels, settings, model=model)

    trained_data = np.array(trained_data, dtype=object)
    os.chdir('training_files')
    np.save(f'training_data_{model}_{size}.npy', trained_data)
    return False

def load_trainingdata(model='tf', size='small'):
    print(f'training_data_{model}_{size}.npy')
    ladung = np.load(f'training_data_{model}_{size}.npy', allow_pickle=True)
    return tuple(ladung)

def trainingdata(settings: dict) -> tuple:
    """

    Returns:
        object: 
    """
    print(settings['model'])
    create_trainingdata(settings)
    return load_trainingdata(settings['model'], settings['size'])

def main(settings):
    print(trainingdata(settings))
    
args: dict[str, Any] = {
    'printing' : True,
    'model'    : 'torch',
    'size'     : 'bienen_1',
}

if __name__=='__main__':
    main(args)

#64 Anfangsneuronen fest machen