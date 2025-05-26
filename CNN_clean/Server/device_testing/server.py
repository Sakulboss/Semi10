import os
import uuid
import requests
import json
import logging

SERVER_URL = 'https://survive.cermann.com/server.php'
UUID_FILE = '../../Neural_Network/device_uuid.txt'

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.DEBUG)

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
    This function sends a GET request to the server to retrieve the layer to be trained.
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

        # Überprüfen Sie den Inhalt der Antwort
        if not response.text:
            logger.error("Leere Antwort vom Server erhalten")
            return None, None
        try:
            data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Ungültige JSON-Antwort vom Server: {response.text}")
            return None, None
        if 'line' in data and 'line_index' in data:
            return data['line'], data['line_index']
        else:
            logger.debug(f'Server-Antwort: {data.get("message", "Keine Zeile gefunden")}')
            return None, None
    except requests.RequestException as e:
        logger.error(f'Fehler während GET-Anfrage: {str(e)}')
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
    payload = {
        'line_index': line_index,
        'result': result,
        'model': 'default_model',  # Fügen Sie hier Ihr tatsächliches Modell ein
        'epoch': 0                 # Fügen Sie hier Ihre tatsächliche Epoch-Nummer ein
    }
    params = {'key': device_uuid}

    try:
        logger.debug(f"POST {SERVER_URL} with payload {json.dumps(payload)}")
        response = requests.post(SERVER_URL, params=params, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.debug('Server response: ' + str(data.get('message', 'No response message')))
    except requests.RequestException as e:
        logger.critical(f'Error during POST: {str(e)}')
    except json.JSONDecodeError as e:
        logger.critical(f'Error decoding JSON response from server: {str(e)}')


def main(logger):
    device_uuid = load_or_create_uuid()
    print(f'Device UUID: {device_uuid}')

    line, line_index = get_next_line(device_uuid, logger)
    if line is not None and line_index is not None:
        send_result(device_uuid, line_index, line, logger)

if __name__ == '__main__':
    main(logger)