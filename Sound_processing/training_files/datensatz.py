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
        os.chdir('..')

def download_small_dataset():
    directory()

    if not os.path.isfile('_animal_sounds.zip'):
        printer('\nPlease wait a couple of seconds ...')
        try:
            wget.download(
                'https://github.com/machinelistening/machinelistening.github.io/blob/master/animal_sounds.zip?raw=true',
                out='_animal_sounds.zip', bar=None)
            printer('animal_sounds.zip downloaded successfully ...')
        except Exception as e:
            printer(f"Error downloading file: {str(e)}")

    else:
        printer('\nFiles already exist!', '\n')

    if not os.path.isdir('_animal_sounds'):
        printer("\nLet's unzip the file ... ")
        assert os.path.isfile('_animal_sounds.zip')
        with zipfile.ZipFile('_animal_sounds.zip', 'r') as f:
            # unzip all files into current folder
            f.extractall('.')
        assert os.path.isdir('_animal_sounds')
        printer("All done :)", '\n')

def download_big_dataset():
    source_folder: str = '_viele_sounds'
    target_base_folder: str = '_viele_sounds_geordnet'
    entries: list = []
    directories: list = []
    directory()
    with open('esc50.csv', 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                entries.append(line.split(','))
    if not os.path.isdir(target_base_folder):
        os.mkdir(target_base_folder)
    for i in range(len(entries)):
        target_folder = os.path.join(target_base_folder, entries[i][3])
        if not os.path.isdir(target_folder):
            os.mkdir(target_folder)
        target_file = os.path.join(target_folder, entries[i][0])
        if not os.path.isfile(target_file):
            source_file = os.path.join(source_folder, entries[i][0])
            shutil.copy(source_file, target_file)
        directories.append(entries[i][3] + '/' + entries[i][0])
    printer(len(directories))

def create_bienen1():
    source_folder: str = os.path.join('_bees', '27-28_April')
    target_base_folder: str = '_bee_sounds'
    categories: list = ['no_event', 'swarm_event']
    entries: list = []
    directories: list = []
    if not os.path.isdir(target_base_folder):
        os.mkdir(target_base_folder)
    count = 0
    for i in range(len(files := os.listdir(source_folder))):
        if files[i].endswith('.wav'):
            entries.append(files[i])
            if entries[count].endswith('17.wav'):
                target_folder = os.path.join(target_base_folder, 'no_event' )
            else:
                target_folder = os.path.join(target_base_folder, 'swarm_event')

            if not os.path.isdir(target_folder):
                os.mkdir(target_folder)

            target_file = os.path.join(target_folder, entries[count])
            if not os.path.isfile(target_file):
                source_file = os.path.join(source_folder, entries[count])
                shutil.copy(source_file, target_file)
            directories.append(target_folder + '/' + entries[count])
            count += 1
    printer(len(directories))

def dataset(settings):
    directory()
    global printing, file_
    printing   = settings.get('printing', None) or printing
    size: str  = settings.get('size', None)     or 'size'
    file_      = settings.get('file_', None)    or file_

    if size == "big":
        if not os.path.isdir('_viele_sounds_geordnet'):
            download_big_dataset()
        dir_dataset: str = '_viele_sounds_geordnet'
    elif size == 'bienen_1':
        if not os.path.isdir('_bee_sounds'):
            create_bienen1()
        dir_dataset: str = '_bee_sounds'
    else:
        download_small_dataset()  # für den kleinen Datensatz
        dir_dataset: str = '_animal_sounds'
    return glob.glob(os.path.join(dir_dataset, '*'))

if __name__ == '__main__':
    print(dataset({}))