import os
import uuid
import requests
import json


SERVER_URL = 'https://survive.cermann.com/server.php'
UUID_FILE = 'device_uuid.txt'


def load_or_create_uuid():
    """
    This function loads the UUID from a file if it exists, otherwise it creates a new UUID and saves it to the file.
    Returns:
        UUID as a string
    """
    if os.path.exists(UUID_FILE):
        with open(UUID_FILE, 'r') as f:
            device_uuid = f.read().strip()
            return device_uuid
    else:
        device_uuid = str(uuid.uuid4())
        with open(UUID_FILE, 'w') as f:
            f.write(device_uuid)
        return device_uuid


def get_next_line(device_uuid, logger):
    """
    This fuction sends a GET request to the server to retrieve the layer to be trained.
    Args:
        device_uuid: The UUID of the device, used as a key for the request.
        logger: The logger for logging.
    Returns:
        The line and line index from the server response.
    """

    params = {'key': device_uuid}
    try:
        logger.debug(f"GET {SERVER_URL} with params {params}")
        response = requests.get(SERVER_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if 'line' in data and 'line_index' in data:
            return data['line'], data['line_index']
        else:
            logger.debug('Server response:', data.get('message', 'No line found'))
            return None, None
    except requests.RequestException as e:
        logger.critical('Error during GET:', str(e))
        return None, None


def send_result(device_uuid, line_index, result, logger):
    """
    This function sends the result of the training back to the server.
    Args:
        device_uuid: The UUID of the device.
        line_index:  The index of the line that was trained.
        result:      The result of the training.
        logger:      The logger for logging.
    Returns:
        None
    """

    headers = {'Content-Type': 'application/json'}
    payload = {'line_index': line_index, 'result': result}
    params = {'key': device_uuid}

    try:
        logger.debug(f"POST {SERVER_URL} with payload {json.dumps(payload)}")
        response = requests.post(SERVER_URL, params=params, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.debug('Server response:', data.get('message', 'No response message'))
    except requests.RequestException as e:
        logger.critical('Error during POST:', str(e))
    except json.JSONDecodeError:
        logger.critical('Error decoding JSON response from server')


def main(logger):
    device_uuid = load_or_create_uuid()
    print(f'Device UUID: {device_uuid}')

    line, line_index = get_next_line(device_uuid, logger)
    if line is not None and line_index is not None:
        send_result(device_uuid, line_index, line, logger)

if __name__ == '__main__':
    main()
