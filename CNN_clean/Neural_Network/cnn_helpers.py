import os
import json
import time
import uuid
import logging


def get_uuid(uuid_file = 'uuid.txt'):
    """
    This function loads the UUID from a file if it exists, otherwise it creates a new UUID and saves it to the file.
    Args:
        uuid_file: The path to the UUID file.
    Returns:
        UUID as a string
    """
    if os.path.exists(uuid_file):
        with open(uuid_file, 'r') as f:
            device_uuid = f.read().strip()
            return device_uuid
    else:
        device_uuid = str(uuid.uuid4())
        with open(uuid_file, 'w') as f:
            f.write(device_uuid)
        return device_uuid


def setup_logging(level=2):
    # Konfiguriere das Logging-Format und Level
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            #logging.FileHandler('training.log'), # output to file
            logging.StreamHandler()                # concole output
        ]
    )
    logging.getLogger('numba.core.byteflow').setLevel(logging.WARNING)
    logging.getLogger('numba.core.interpreter').setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def load_args(path=None) -> dict:
    """
    This function loads the arguments from a file. The file is created in the create_trainingdata function. The correct file is found by the size and model.
    Returns:
        contents of the file as a dictionary
    """
    if path is None:
        path = os.path.join(os.getcwd(), 'config.json')

    with open(path, 'r') as file:
        args = json.load(file)
    return args


def animate():
    animation = "|/-\\"
    idx = 0
    while True:
        print('Netz wird trainiert ', animation[idx % len(animation)], end="\r")
        idx += 1
        time.sleep(0.1)
