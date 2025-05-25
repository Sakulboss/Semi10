import os
import logging
import shutil
import glob

from tqdm import tqdm


def setup_logging(args: dict) -> logging.Logger:
    handlers = []
    if args.get('log_to_file', False):   logging.FileHandler(args.get('log_file', 'training.log'))
    if args.get('log_to_console', True): handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=args.get('level', 2),
        format=args.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=handlers
    )
    return logging.getLogger(__name__)


def change_cwd_to_training_files(logging_args):
    logger = setup_logging(logging_args)

    os.chdir('..')
    os.chdir('files')
    os.chdir('_esc50')
    logger.debug(f'Set training files storage location to: {os.getcwd()}')


def use_esc50(args):
    """
    This function categorizes the ESC-50 dataset. It was only used to check whether CNN could work
    with big datasets. CNN was not trained on this dataset.
    Returns:
        None
    """

    source_folder: str = args.get('training_files_storage_location', os.path.join('data', ''))
    target_base_folder: str = '_viele_sounds_geordnet'
    entries: list = []
    directories: list = []
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


def create_bee_1(args: dict = None) -> None:
    """
    This function categorizes the bee sounds captured by us. The KNN was trained on this dataset.
    Args:
        args: dictionary with the settings for the dataset like file storage locations, etc.
    Returns: None

    """
    ext = args.get("training_file_extensions","wav")
    source_folder: str = args.get("training_files_storage_location", os.path.join(os.getcwd(),'_bees\\27-28_April'))
    target_base_folder: str = os.path.join(args.get("sorted_files_storage_location", os.getcwd()), '_bee_sounds')
    entries: list = []
    directories: list = []
    if not os.path.isdir(target_base_folder):
        os.mkdir(target_base_folder)
    count = 0
    for i in tqdm(range(len(files := os.listdir(source_folder)))):
        if files[i].endswith(ext):
            entries.append(files[i])
            if entries[count].endswith(f'17.{ext}'):
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


def dataset(size: str, args: dict, logging_args) -> list[str]:
    """
    This function defines the used dataset. Therefor the working directory is changed to the correct one. It then creates the sorted dataset if it doesn't exist yet. The dataset is then returned as a list of paths.

    Args:
        size: string, size of the dataset ('esc50' or 'bienen_1')
        args: dictionary with the settings for the dataset like file storage locations, etc.
        logging_args: get arguments for logging
    Returns:
        list[str]: list of paths to the dataset
    """
    logger = setup_logging(logging_args)



    if size == "esc50":
        """
        Wichtig -> muss gefixed werden
        """
        sorted_files = os.path.join(args.get("sorted_files_storage_location", os.getcwd()), '_esc50')
        if not os.path.isdir(sorted_files):
            use_esc50(args)
        dir_dataset: str = '_esc50_sorted'
    elif size == 'bees_1':
        sorted_files = os.path.join(args.get("sorted_files_storage_location", os.getcwd()), '_bee_sounds')
        #if not os.path.isdir(sorted_files):
        create_bee_1(args)
        dir_dataset: str = sorted_files
    else:
        raise ValueError('Invalid dataset size.')

    logger.critical('ESC50 Fixen!!! Umsetzung von Zielspeicherort für ZIP notwendig, wenn möglich auch unsere Daten als Zip zum herunterladen parat haben und das einbauen')
    logger.warning(glob.glob(os.path.join(dir_dataset, '*')))
    return glob.glob(os.path.join(dir_dataset, '*'))

