import os
import sys

import wget
import zipfile
import shutil
import glob

printing: bool = False
file_ = sys.stdout

def printer(*text, sep=' ', end='\n', file=None):
    global printing
    global file_
    file = file or file_
    if printing: print(*text, sep=sep, end=end, file=file)

def directory():
    paths = os.getcwd()
    if not os.path.basename(paths) == 'Sound_processing':
        os.chdir('../Sound_processing')

def download_dataset(printing=False):
    directory()

    if not os.path.isfile('../Sound_processing/animal_sounds.zip'):
        printer('\nPlease wait a couple of seconds ...')
        try:
            wget.download(
                'https://github.com/machinelistening/machinelistening.github.io/blob/master/animal_sounds.zip?raw=true',
                out='_animal_sounds.zip', bar=None)
            printer('_animal_sounds.zip downloaded successfully ...')
        except Exception as e:
            printer(f"Error downloading file: {str(e)}")

    else:
        printer('\nFiles already exist!', '\n')

    if not os.path.isdir('_animal_sounds'):
        printer("\nLet's unzip the file ... ")
        assert os.path.isfile('../Sound_processing/animal_sounds.zip')
        with zipfile.ZipFile('../Sound_processing/animal_sounds.zip', 'r') as f:
            # unzip all files into current folder
            f.extractall('.')
        assert os.path.isdir('_animal_sounds')
        printer("All done :)", '\n')


def download_big_dataset(printing=False):
    source_folder: str = '_viele_sounds'
    target_base_folder: str = '_viele_sounds_geordnet'
    eintraege: list = []
    directories: list = []

    directory()

    with open('esc50.csv', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                eintraege.append(line.split(','))
    if not os.path.isdir(target_base_folder):
        os.mkdir(target_base_folder)
    for i in range(len(eintraege)):
        target_folder = os.path.join(target_base_folder, eintraege[i][3])
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)
        target_file = os.path.join(target_folder, eintraege[i][0])
        if not os.path.isfile(target_file):
            source_file = os.path.join(source_folder, eintraege[i][0])
            shutil.copy(source_file, target_file)
        directories.append(eintraege[i][3] + '/' + eintraege[i][0])
    printer(len(directories))

def dataset(setting: dict):
    directory()
    global printing, file_
    printing = setting.get('print', False)
    file_ = setting.get('file', sys.stdout)

    big: bool = setting.get('big', False)

    if big:
        download_big_dataset(printing=printing)
        dir_dataset: str = '_viele_sounds_geordnet'
    else:
        download_dataset(printing=printing)  # für den kleinen Datensatz
        dir_dataset: str = '_animal_sounds'
    return glob.glob(os.path.join(dir_dataset, '*'))


setting: dict = {
    'big': False,
    'print': True
}

if __name__ == '__main__':
    print(dataset(setting))