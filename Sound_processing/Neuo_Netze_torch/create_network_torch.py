import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from Sound_processing.training_files.driver_mels import trainingdata
import numpy as np

args = {
    'printing'         : True,
    'size'             : 'big',
    'plot_history'     : False,
    'confusion_matrix' : False,
    'model'            : 'torch',
}


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
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


r'''
# Define the model
    model = Sequential()
    model.add(Input(shape=input_shape))
    # 1st Convolutional Layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # Flatten the output to feed into the Dense layer
    model.add(Flatten())

    # Output layers for classification
    model.add(Dense(units=64, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))
    model.add(Dense(units=n_classes, activation='softmax'))'''

class CustomDataset(Dataset):
    def __init__(self, features, labels, train = True):
        super(CustomDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def check_accuracy(loader, model):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
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

            num_correct += (predictions_new == y_new).sum()  #Count correct predictions
            num_samples += predictions.size(0)  #Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        if loader.dataset.train:
            print(f"train: Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
        else:
            print(f"test: Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")


    model.train()  # Set the model back to training mode


device = "cuda" if torch.cuda.is_available() else "cpu"

input_size    = 64 * 100
num_classes   = 50 # 5 small, 50 big
learning_rate = 0.001
batch_size    = 64
num_epochs    = 10

data = trainingdata(args)
X_train_norm = data[2]
y_train_transformed = data[3]
X_test_norm = data[4]
y_test_transformed = data[6]

X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_transformed, dtype=torch.float)
X_test_tensor = torch.tensor(X_test_norm, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_transformed, dtype=torch.float)

train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor, train=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = CNN(in_channels=1, output_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
    for batch_index, (data, targets) in enumerate(tqdm(train_loader, disable=True)):
        # Move data and targets to the device (GPU/CPU)
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass: compute the model output
        scores = model(data)

        #print(scores.shape, targets.shape)
        #print(scores, targets)
        loss = criterion(scores, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()
    check_accuracy(test_loader, model)

check_accuracy(train_loader, model)