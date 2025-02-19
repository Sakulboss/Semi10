import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from Sound_processing.Neuo_Netze_torch.haupt_driver_torch import main
import numpy as np

args = {
    'printing'         : True,
    'big'              : False,
    'plot_history'     : True,
    'confusion_matrix' : True,
}


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For MNIST, this is 1 (grayscale images).
            num_classes: int
                The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(CNN, self).__init__()

        # First convolutional layer: 1 input channel, 8 output channels, 3x3 kernel, stride 1, padding 1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3,3), stride=1, padding=1)
        # Max pooling layer: 2x2 window, stride 2
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        # Second convolutional layer: 8 input channels, 16 output channels, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=1, padding=1)
        # Fully connected layer: 16*7*7 input features (after two 2x2 poolings), 10 output features (num_classes)
        self.fc1 = nn.Linear(64 * 100, num_classes)

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
        x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
        x = self.pool(x)           # Apply max pooling
        x = x.view(x.size(0), -1)
        #x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        x = self.fc1(x)            # Apply fully connected layer
        return x


def check_accuracy(loader, model):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """
    #if loader.dataset.train:
        #print("Checking accuracy on training data")
    #else:
        #print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y = np.argmax(y, axis=1)
            # Forward pass: compute the model output
            _, predictions = model(x).max(1)  # Get the index of the max log-probability
            print('predictions',predictions)
            print('test', y)

            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    model.train()  # Set the model back to training mode


device = "cuda" if torch.cuda.is_available() else "cpu"

input_size = 64 * 100  # 28x28 pixels (not directly used in CNN)
num_classes = 5  # digits 0-9
learning_rate = 0.001
batch_size = 64
num_epochs = 10  # Reduced for demonstration purposes

#train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())
#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

#test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Custom dataset class

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        super(CustomDataset, self).__init__()
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets
data = main(args)
X_train_norm = data[2]
y_train_transformed = data[3]
X_test_norm = data[4]
y_test_transformed = data[6]

X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_transformed, dtype=torch.float)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_transformed, dtype=torch.float)

train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

'''
data = main(args)
X_train_tensor = torch.tensor(X_train_norm:= data[2], dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_transformed:=data[3], dtype=torch.long)
print(X_train_tensor.shape)
print(y_train_tensor.shape)
X_test_tensor = torch.tensor(X_test_norm:=data[4], dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_transformed:=data[6], dtype=torch.long)'''


model = CNN(in_channels=1, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for batch_index, (data, targets) in enumerate(tqdm(train_loader)):
        # Move data and targets to the device (GPU/CPU)
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass: compute the model output
        scores = model(data)

        print(scores.shape, targets.shape)
        print(scores, targets)
        loss = criterion(scores, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()

print(check_accuracy(train_loader, model))
print(check_accuracy(test_loader, model))


#bisschen Code halt


r'''

Epoch [1/10]
  0%|          | 0/3 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "C:\Users\sfz-a\PycharmProjects\Semi10\Sound_processing\malte\create_network.py", line 170, in <module>
    loss = criterion(scores, targets)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sfz-a\Documents\GitHub\Semi10\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sfz-a\Documents\GitHub\Semi10\.venv\Lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\sfz-a\Documents\GitHub\Semi10\.venv\Lib\site-packages\torch\nn\modules\loss.py", line 1295, in forward
    return F.cross_entropy(
           ^^^^^^^^^^^^^^^^
  File "C:\Users\sfz-a\Documents\GitHub\Semi10\.venv\Lib\site-packages\torch\nn\functional.py", line 3494, in cross_entropy
    return torch._C._nn.cross_entropy_loss(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: 0D or 1D target tensor expected, multi-target not supported


'''