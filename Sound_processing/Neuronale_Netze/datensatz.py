import os
import wget
import zipfile
import shutil
import glob

def directory():
    paths = os.getcwd()
    if not os.path.basename(paths) == 'Sound_processing':
        os.chdir('..')



def download_dataset(printing=False):
    if not os.path.isfile('animal_sounds.zip'):
        if printing: print('\nPlease wait a couple of seconds ...')
        try:
            wget.download(
                'https://github.com/machinelistening/machinelistening.github.io/blob/master/animal_sounds.zip?raw=true',
                out='animal_sounds.zip', bar=None)
            if printing: print('animal_sounds.zip downloaded successfully ...')
        except Exception as e:
            if printing: print(f"Error downloading file: {str(e)}")

    else:
        if printing: print('\nFiles already exist!', '\n')

    if not os.path.isdir('animal_sounds'):
        if printing: print("\nLet's unzip the file ... ")
        assert os.path.isfile('animal_sounds.zip')
        with zipfile.ZipFile('animal_sounds.zip', 'r') as f:
            # unzip all files into current folder
            f.extractall('.')
        assert os.path.isdir('animal_sounds')
        if printing: print("All done :)", '\n')

def download_big_dataset(printing=False):
    source_folder = 'viele_sounds'
    target_base_folder = 'viele_sounds_geordnet'
    eintraege = []
    directories = []
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
    if printing: print(len(directories))

def dataset(**kwargs):
    big = kwargs.get('big', False)
    printing = kwargs.get('printing', False)

    if big:
        download_big_dataset(printing=printing)
        dir_dataset = 'viele_sounds_geordnet'
    else:
        download_dataset(printing=printing)  # für den kleinen Datensatz
        dir_dataset = 'animal_sounds'
    return glob.glob(os.path.join(dir_dataset, '*'))

if __name__ == '__main__':
    print(dataset(big=True))