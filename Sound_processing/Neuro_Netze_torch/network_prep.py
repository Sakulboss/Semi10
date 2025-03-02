import torch
import torch.nn.functional as f
from torch import nn
import numpy as np


class CNN(nn.Module):
    def __init__(self, in_channels, output_classes=5):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For MNIST, this is 1 (grayscale images).
            output_classes: int
                The number of classes we want to predict, in our case 5 (digits 0 to 4).
        """
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 100, output_classes)

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
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x




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

    with torch.no_grad():
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
            print(f"test: Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")


    model.train()  # Set the model back to training mode
