# -*- coding: utf-8 -*-
import requests
import torch.nn.functional as f
from torch import nn
import os

from cnn_helpers import get_uuid


def split_list(lst, delimiter):
    result = []
    current = []
    for item in lst:
        if item == delimiter:
            result.append(current)
            current = []
        else:
            current.append(item)
    result.append(current)  # Letzten Abschnitt hinzufügen
    return result


def move_working_directory():
    working_directory = os.getcwd()
    os.chdir('..')
    os.chdir('files')


def get_next_line(server_url, logger, uuid_file_path=None):
    """
    This fuction sends a GET request to the server to retrieve the layer to be trained.
    Args:
        server_url:      The URL of the server to send the request to.
        logger:          The logger for logging.
        uuid_file_path:  The path to the uuid file to use.
    Returns:
        The line and line index from the server response.
    """

    params = {'key': get_uuid(uuid_file_path)}
    try:
        logger.debug(f"GET {server_url} with params {params}")
        response = requests.get(server_url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'line' in data and 'line_index' in data:
            return data['line'], data['line_index']
        else:
            logger.error('Server response:', data.get('message', 'No line found'))
            return None, None
    except requests.RequestException as e:
        logger.critical('Error during GET:', str(e))
        return None, None


def getnextmodel(file_path: str) -> str | None:
    """
    Args:
        file_path: The path to the file with the model structures.
    Returns:
        model structure
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.startswith('- '):
            lines[i] = '#' + line[1:]
            position = i
            break
    else:
        return None
    with open(file_path, 'w') as file:
        file.writelines(lines)
    return lines[position][2:]


def create_pooling_layer(layer_description: str):
    parts = layer_description.split(';')
    pool_type = parts[1].strip()
    kernel_size = tuple(map(int, parts[2].strip().strip('()').split(',')))
    stride = int(parts[3].strip())
    padding = tuple(map(int, parts[4].strip().strip('()').split(',')))
    if pool_type == 'maxpool':
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    elif pool_type == 'avgpool':
        return nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    else:
        raise ValueError(f"Unbekannter Pooling-Typ: {pool_type}")


def create_conv_layer(layer_description: str):
    parts = layer_description.split(';')
    channels = tuple(map(int, parts[2].strip().strip('()').split(',')))
    kernel_size = tuple(map(int, parts[3].strip().strip('()').split(',')))
    stride = int(parts[4].strip())
    padding = tuple(map(int, parts[5].strip().strip('()').split(',')))
    return nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, stride=stride, padding=padding)


def create_linear_layer(layer_description):
    parts = layer_description.split(';')
    channels = tuple(map(int, parts[2].strip().strip('()').split(',')))
    return nn.Linear(in_features=channels[0], out_features=channels[1])


def reshape_tensor(tensor):
    return tensor.view(tensor.shape[0], -1)


def getlayers(logger, args:dict):

    path:str           = args.get('model_structure_file', os.path.join(os.getcwd(), '_netstruct.txt'))
    server_url:str     = args.get('server_url', 'https://survive.cermann.com/server.php')
    uuid_file_path:str = args.get('uuid_file_path', 'device_uuid.txt')
    use_server:bool    = args.get('use_server', True)
    omt:str            = args.get('model_structure_text', '')
    training_once      = args.get('train_once', True)

    move_working_directory()
    if training_once:
        original_model_text = omt
    elif use_server:
        print(f"Using server: {server_url}")
        original_model_text, line = get_next_line(server_url, logger, uuid_file_path)
    else:
        original_model_text = getnextmodel(path)

    if original_model_text is None:
        return None, None

    layers = original_model_text.split(';;')
    functions = []
    for layer in layers:
        layer = layer.strip()
        if layer.startswith('l'):
            if 'conv2d' in layer:
                functions.append(create_conv_layer(layer))
            elif 'linear' in layer:
                functions.append(create_linear_layer(layer))
            else:
                print(f'Error: Layer type not found: --{layer}')
        elif layer.startswith('p'):
            functions.append(create_pooling_layer(layer))
        elif layer.startswith('a'):
            if 'sigmoid' in layer:
                functions.append(f.sigmoid)
            elif 'relu' in layer:
                functions.append(f.relu)
            elif 'tanh' in layer:
                functions.append(f.tanh)
            else:
                print("Error: Activation function not found: --{}".format(layer))
        elif layer.startswith('v'):
            functions.append(reshape_tensor)
        elif layer == '':
            pass
        else:
            print(f'Error: Layer class not found: --{layer}')

    if use_server and not training_once:
        return functions, original_model_text, line
    else:
        return functions, original_model_text

#stride:     how the filter moves
#padding:    frame for the old picture added
#diletation: not needed, but it adds space between filter kernels (pure brainfuck)


class CNN(nn.Module):
    def __init__(self, logger, args:dict):
        """
        Parameters:
            logger: The logger for logging.
            args:   The settings for the neural network


        If not given, the structure is read from the _netstruct.txt file.
        """

        super(CNN, self).__init__()

        if args.get('use_server', True) and not args.get('train_once', True):
            layers, text, self.line = getlayers(logger, args)
        else:
            layers, text = getlayers(logger, args)
            self.line = None

        working = True

        if (layers or text) is None:
            working = False

        self.text = text
        self.working = working

        self.module_layers = nn.ModuleList(
            [layer for layer in layers if isinstance(layer, nn.Module)]
        )
        self.functions = [
            layer for layer in layers if callable(layer) and not isinstance(layer, nn.Module)
        ]

        self.layers = layers


    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """

        layer_count = 0
        func_count = 0

        for layer in self.layers:
            if isinstance(layer, nn.Module):
                x = self.module_layers[layer_count](x)
                layer_count += 1
            elif callable(layer):
                x = self.functions[func_count](x)
                func_count += 1
            else:
                print(f"Error: Unknown layer type: {layer}")
        return x


    def __str__(self):
        """
        Returns start text of the neural network.
        """
        if self.text.endswith('\n'):
            self.text = self.text[:-2]
        return self.text, self.line

