import os
import uuid
import requests
import json

# Configuration
SERVER_URL = 'https://survive.cermann.com/server.php'  # Change this to your server PHP URL
UUID_FILE = 'device_uuid.txt'

def load_or_create_uuid():
    if os.path.exists(UUID_FILE):
        with open(UUID_FILE, 'r') as f:
            device_uuid = f.read().strip()
            return device_uuid
    else:
        device_uuid = str(uuid.uuid4())
        with open(UUID_FILE, 'w') as f:
            f.write(device_uuid)
        return device_uuid

def get_first_line(device_uuid):
    params = {'key': device_uuid}
    try:
        print(f"GET {SERVER_URL} with params {params}")
        response = requests.get(SERVER_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if 'line' in data and 'line_index' in data:
            return data['line'], data['line_index']
        else:
            print('Server response:', data.get('message', 'No line found'))
            return None, None
    except requests.RequestException as e:
        print('Error during GET:', str(e))
        return None, None

def send_result(device_uuid, line_index, result):
    print(line_index)
    headers = {'Content-Type': 'application/json'}
    payload = {'line_index': line_index, 'result': result}
    params = {'key': device_uuid}
    try:
        print(f"POST {SERVER_URL} with payload {json.dumps(payload)}")
        response = requests.post(SERVER_URL, params=params, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        print('Server response:', data.get('message', 'No response message'))
    except requests.RequestException as e:
        print('Error during POST:', str(e))
    except json.JSONDecodeError:
        print('Error decoding JSON response from server')

def main():
    device_uuid = load_or_create_uuid()
    print(f'Device UUID: {device_uuid}')

    line, line_index = get_first_line(device_uuid)
    if line is not None and line_index is not None:
        print('Received line:', line)
        # Example processing: convert line to uppercase as result
        result = line.upper()
        send_result(device_uuid, line_index, result)
    print("Clean up the files!!! Don't fill them with garbage!!!")

if __name__ == '__main__':
    main()
