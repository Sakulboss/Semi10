import numpy as np
import torch
from torch import nn, no_grad
import torch.nn.functional as f
from torch import nn
import os

def getnextmodel(file_path: str) -> str | None:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.startswith('- '):
            lines[i] = '# ' + line[1:]
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

def getlayers():
    original_model_text = getnextmodel('_netstruct.txt')
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
            functions.append('view')
        elif layer == '':
            pass
        else:
            print(f'Error: Layer class not found: --{layer}')
    return functions


#stride:     how the filter moves
#padding:    frame for the old picture added
#diletation: not needed, but it adds space between filter kernels (pure brainfuck)

class CNN(nn.Module):
    def __init__(self):
        """
        Parameters:
        No given parameters. Net is created with the given structure in the _netstruct.txt file.
        """

        super(CNN, self).__init__()
        self.layers : list[str|nn.Module] = getlayers()

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
        for layer in self.layers:
            if isinstance(layer, nn.Module):
                x = layer(x)
            elif layer.startswith('v'):
                x = x.view(x.size(0), -1)
            elif layer == '':
                pass
            else:
                print('Error: Layer type not found')

        return x



