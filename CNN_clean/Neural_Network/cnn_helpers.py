import os
import uuid


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



