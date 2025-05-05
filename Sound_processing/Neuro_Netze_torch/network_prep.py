import numpy as np
import torch
from torch import nn, no_grad
import torch.nn.functional as f
from torch import nn

import os

from torchgen.executorch.api.et_cpp import return_type

r"""
l; layertype;        channels (in, out); kernel_size (h, w); stride; padding;;
l; conv2d, linear;   (3,3);             (3,3);               1;      1;;
p; pooltype;         size (h, w);       stride;              padding;;
p; avgpool, maxpool; (2,2);             1;                   0;;
a; activation_funtion;;
a; sigmoid, relu, tanh;;
v; view;;
"""

def getnextmodel(file_path: str) -> str | None:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.startswith('-'):
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

def getlayers():
    original_model_text = getnextmodel('netstruct.txt')
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

print(getnextmodel('netstruct.txt'))
print(getlayers())




#stride:  how the filter moves
#padding: frame for the old picture added
#diletation: not needed, but it adds space between filter kernels (pure brainfuck)

class CNN(nn.Module):
    def __init__(self):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. Because we use only calculated two-dimensional frames, we have only one channel.
            output_classes: int
                The number of classes we want to predict, in our case 2.
        """

        super(CNN, self).__init__()


        """
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(2,2), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 100, output_classes)"""


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
            elif layer.startswith('v'): #-----Fertig
                x = x.view(x.size(0), -1)
            elif layer == '':
                pass
            else:
                print('Error: Layer type not found')

        '''
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)'''
        return x



'Error: Activation function not found'


def check_accuracy(loader, model, device):

    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        device: string
            The Device to run the model on.
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """


    num_correct = 0
    num_samples = 0
    model.eval()

    with no_grad():
        for x, y in loader:

            x = x.to(device)
            _, predictions = model(x).max(1)

            predictions_new = np.array(predictions.cpu())

            if loader.dataset.train:
                y_new = np.array(y.max(1))
                y_new = np.array([int(i) for i in y_new[1]])
            else:
                y_new = np.array([int(i) for i in y])

            num_correct += (predictions_new == y_new).sum()
            num_samples += predictions.size(0)

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        if loader.dataset.train:
            print(f"train: Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
        else:
            print(f"test:  Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")


    model.train()  # Set the model back to training mode
