import os
import sys
import wget
import zipfile
import shutil
import glob


def directory():
    """
    This function changes the working directory. It is necessary so that the files and directories are created and found correctly.
    Returns:
        None
    """
    paths = os.getcwd()
    if not os.path.basename(paths) == 'Sound_processing':
        os.chdir('..')


def download_small_dataset() -> None:
    """
    This function downloads, unzips and categorizes a small dataset from our external supervisor. It was only used for testing purposes. The KNN was not trained on this dataset.
    Returns:
        None
    """
    directory()
    if not os.path.isfile('_animal_sounds.zip'):
        try:
            wget.download(
                'https://github.com/machinelistening/machinelistening.github.io/blob/master/animal_sounds.zip?raw=true',
                out='_animal_sounds.zip', bar=None)
        except Exception as e:
            print(f"Error downloading file: {str(e)}")
    if not os.path.isdir('_animal_sounds'):
        assert os.path.isfile('_animal_sounds.zip')
        with zipfile.ZipFile('_animal_sounds.zip', 'r') as f:
            f.extractall('.')
        assert os.path.isdir('_animal_sounds')


def download_big_dataset():
    """
    This function categorizes the ESC-50 dataset. It was only used to check whether the KNN could work with big datasets. The KNN was not trained on this dataset.
    Returns:
        None
    """
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


def create_bienen1():
    """
    This function categorizes the bee sounds captured by us. The KNN was trained on this dataset.
    Returns:

    """
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


def dataset(size: str) -> list[str]:
    """
    This function defines the used dataset. Therefor the working directory is changed to the correct one. It then creates the sorted dataset if it doesn't exist yet. The dataset is then returned as a list of paths.

    Args:
        size: string, size of the dataset ('small', 'big', 'bienen_1')

    Returns:
        list[str]: list of paths to the dataset
    """
    directory()

    if size == "big":
        if not os.path.isdir('_viele_sounds_geordnet'):
            download_big_dataset()
        dir_dataset: str = '_viele_sounds_geordnet'
    elif size == 'bienen_1':
        if not os.path.isdir('_bee_sounds'):
            create_bienen1()
        dir_dataset: str = '_bee_sounds'
    else:
        if not os.path.isdir('_animal_sounds'):
            download_small_dataset()
        dir_dataset: str = '_animal_sounds'
    return glob.glob(os.path.join(dir_dataset, '*'))

