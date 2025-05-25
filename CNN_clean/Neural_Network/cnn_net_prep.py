# -*- coding: utf-8 -*-
import requests
import torch.nn.functional as f
from torch import nn
import os
import logging
import torch
import uuid


# Hints for the layers:
# stride:     how the filter moves
# padding:    frame for the old picture added
# diletation: not needed, but it adds space between filter kernels (pure brainfuck)


def move_working_directory(target:str ='files') -> None:
    """
    This function changes the current working directory (cwd) to the specified target directory.
    Args
        target: str target directory to change to, default is 'files'
    Returns:
        None
    """
    os.chdir('..')
    os.chdir(target)


def get_uuid(uuid_file:str = 'device_uuid.txt') -> str:
    """
    This function loads the UUID from a file if it exists, otherwise it creates a new UUID and saves it to the file.
    Args:
        uuid_file: str path to the UUID file
    Returns:
        UUID as a string
    """
    move_working_directory()
    if os.path.exists(uuid_file):
        with open(uuid_file, 'r') as file:
            device_uuid = file.read().strip()
            return device_uuid
    else:
        device_uuid = str(uuid.uuid4())
        with open(uuid_file, 'w') as file:
            file.write(device_uuid)
        return device_uuid


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


def get_next_line(server_url, logger, uuid_file_path=None):
    """
    This fuction sends a GET request to the server to retrieve the layer to be trained.
    Args:
        server_url:     str            The URL of the server to send the request to.
        logger:         logging.Logger The logger for logging.
        uuid_file_path: str            The path to the uuid file to use - if not given, the uuid will be created new.
    Returns:
        The line and line index from the server response. The line index is used only on the server to change the status of the trained line.
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
    Ths function reads the model structure from the given file.
    Args:
        file_path: str path to the file with the model structures
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


def create_pooling_layer(layer_description: str) -> nn.MaxPool2d | nn.AvgPool2d:
    """
    Creates a pooling layer based on the provided layer description.
    Args:
        layer_description: str description for pooling layer
    Returns:
        nn.MaxPool2d: a pooling layer, built like layer_description
    """
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


def create_conv_layer(layer_description: str) -> nn.Conv2d:
    """
    Creates a convolutional layer based on the provided layer description.
    Args:
        layer_description: str description for convolutional layer
    Returns:
        nn.Conv2d: a convolutional layer, built like layer_description
    """
    parts = layer_description.split(';')
    channels = tuple(map(int, parts[2].strip().strip('()').split(',')))
    kernel_size = tuple(map(int, parts[3].strip().strip('()').split(',')))
    stride = int(parts[4].strip())
    padding = tuple(map(int, parts[5].strip().strip('()').split(',')))
    return nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, stride=stride, padding=padding)


def create_linear_layer(layer_description: str) -> nn.Linear:
    """
    Args:
        layer_description: str description for linear layer
    Returns:
        nn.Linear: a linear layer, built like layer_description
    """
    parts = layer_description.split(';')
    channels = tuple(map(int, parts[2].strip().strip('()').split(',')))
    return nn.Linear(in_features=channels[0], out_features=channels[1])


def getlayers(logger: logging.Logger, args:dict) -> tuple:
    """
    This function prepares the layers of the neural network. It reads the model structure from a file or from a server and creates the layers based on the model text.
    Args:
        logger: logging.Logger The logger for logging.
        args:   dict           The arguments for the neural network and server.
    Returns:
        list containing working layers, original model text and line index
    """
    path:str           = args.get('model_structure_file', os.path.join(os.getcwd(), '_netstruct.txt'))
    server_url:str     = args.get('server_url', 'https://survive.cermann.com/server.php')
    uuid_file_path:str = args.get('uuid_file_path', 'device_uuid.txt')
    use_server:bool    = args.get('use_server', True)
    omt:str            = args.get('model_text', '')
    training_once      = args.get('train_once', True)

    move_working_directory()

    # Get the right model text and set the line index
    if training_once:
        original_model_text = omt
        line = -1
    elif use_server:
        original_model_text, line = get_next_line(server_url, logger, uuid_file_path)
    else:
        original_model_text = getnextmodel(path)
        line = -1

    if original_model_text is None:
        return None, None, None

    layers = original_model_text.split(';;')
    functions = []

    # Split the text into working layers with some fancy logic (just big if statements)
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
                functions.append(f.gelu)
            elif 'tanh' in layer:
                functions.append(f.tanh)
            else:
                print("Error: Activation function not found: --{}".format(layer))
        elif layer.startswith('v'):
            reshape_tensor = lambda tensor: tensor.view(tensor.shape[0], -1)
            functions.append(reshape_tensor)
        elif layer == '':
            pass
        else:
            print(f'Error: Layer class not found: --{layer}')

    return functions, original_model_text, line


class CNN(nn.Module):
    def __init__(self, logging_args: dict, args:dict):
        """
        Initializes the CNN model with the given logging arguments and settings.
        Parameters:
            logging_args: dict The arguments for logging.
            args:         dict The settings for the neural network
        """
        # Initialize variables
        super(CNN, self).__init__()
        logger = setup_logging(logging_args)
        layers, text, line = getlayers(logger, args)
        self.line = line
        working = True

        if (layers or text) is None:
            working = False

        logger.debug(f"Model structure: {layers}")
        logger.debug(f"Model structure text: {text}")
        logger.debug(f"Model structure line: {line}")

        self.text:str        = text
        self.working:bool    = working
        self.accuracy:list   = []
        self.se:list         = []
        self.epoch_time:list = []
        self.scrore:int      = -1
        self.epoch_max:int   = 1


        # Split the layers into module layers and functions, needed for torch to remember the weights in the layers
        self.module_layers = nn.ModuleList(
            [layer for layer in layers if isinstance(layer, nn.Module)]
        )
        self.functions = [
            layer for layer in layers if callable(layer) and not isinstance(layer, nn.Module)
        ]

        self.layers = layers


    def forward(self, x) -> torch.Tensor:
        """
        This function defines the forward pass of the CNN.
        Parameters:
            x: torch.Tensor input tensor
        Returns:
            The through the net passed tensor
        """

        layer_count = 0
        func_count = 0

        # Some logic to undo the splitting in the __init__ method, it will create the network structure
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

    def __int__(self):
        """
        Returns the line index from the server response.
        """
        return self.line


    def se_func(self, se=None):
        """
        Sets or returns the squared error of the neural network in training.
        """
        if se is None:
            return self.mse
        self.se.append(se)

    def epoch(self, epoch=None):
        """
        Sets or returns the epoch of the neural network.
        """
        if epoch is None:
            return self.epoch_max
        self.epoch_max = epoch


    def __str__(self):
        """
        Returns start text of the neural network.
        """
        if self.text.endswith('\n'):
            self.text = self.text[:-2]
        return self.text

