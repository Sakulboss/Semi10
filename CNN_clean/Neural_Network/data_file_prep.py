import os, logging, shutil, glob
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


def use_esc50(args:dict) -> str:
    """
    This function categorizes the ESC-50 dataset. It was only used to check whether CNN could work
    with big datasets. CNN was not trained on this dataset.
    Args:
        args: dict with the settings for the dataset like file storage locations, etc.
    Returns:
        path with the sorted files
    """
    source_folder: str = args.get('training_files_storage_location', os.path.join(os.getcwd(), '_esc50'))
    target_base_folder: str = os.path.join(args.get('sorted_files_storage_location', os.getcwd()), '_esc50_sorted')
    entries: list = []
    directories: list = []
    with open(args.get('esc50_file','esc50.csv'), 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
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
    return target_base_folder

def create_bee_1(args: dict = None) -> str:
    """
    This function categorizes the bee sounds captured by us. The KNN was trained on this dataset.
    Args:
        args: dict with the settings for the dataset like file storage locations, etc.
    Returns:
        path with the sorted files
    """
    ext = args.get("training_file_extensions","wav")
    source_folder: str = args.get("training_files_storage_location", os.path.join(os.getcwd(),'_bees/27-28_April'))
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
    return target_base_folder


def dataset(size: str, logging_args: dict, args: dict) -> list[str]:
    """
    This function defines the used dataset. Therefor the working directory is changed to the correct one. It then creates the sorted dataset if it doesn't exist yet. The dataset is then returned as a list of paths.
    Args:
        size:         str  size of the dataset ('esc50' or 'bees_1')
        logging_args: dict with arguments for logging
        args:         dict with main settings for the dataset like file storage locations, etc.
    Returns:
        list of paths to the dataset
    """
    logger = setup_logging(logging_args)
    new_folder = args.get("create_new_source", True)
    if size == "esc50":
        sorted_files = os.path.join(args.get("sorted_files_storage_location", os.getcwd()), '_esc50_sorted')
        if new_folder or not os.path.isdir(sorted_files):
            sorted_files = use_esc50(args)
        dir_dataset: str = sorted_files
    elif size == 'bees_1':
        sorted_files = os.path.join(args.get("sorted_files_storage_location", os.getcwd()), '_bee_sounds')
        if new_folder or not os.path.isdir(sorted_files):
            sorted_files = create_bee_1(args)
        dir_dataset: str = sorted_files
    else:
        raise ValueError('Invalid dataset size.')

    logger.warning(f'Using the following directories: {glob.glob(os.path.join(dir_dataset, "*"))}')
    return glob.glob(os.path.join(dir_dataset, '*'))

