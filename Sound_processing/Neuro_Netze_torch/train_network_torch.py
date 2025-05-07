import torch
from torch import optim, no_grad, nn
from tqdm import tqdm
import os
import numpy as np

from Sound_processing.Neuro_Netze_torch.network_prep import CNN

def move_working_directory():
    working_directory = os.getcwd()
    for i in range(3):
        if os.path.basename(working_directory) != "Sound_processing":
            os.chdir('..')
            break
    os.chdir('modelle')

def get_new_filename(file_extension: str) -> str:
    move_working_directory()
    count = len([counter for counter in os.listdir(os.getcwd()) if counter.endswith(file_extension)]) + 1
    return f'model_torch_{count}.{file_extension}'

def train(loader, args):

    train_loader = loader[0]
    test_loader = loader[1]
    num_epochs = args.get('epochs', 10)
    learning_rate = args.get('learning_rate')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accuracy = []
    epoch_max = num_epochs
    model = CNN()
    if model.working is False:
        return None
    model = model.to(device=device)
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
        accuracy.append((1-check_accuracy(test_loader, model, device))**2)
        if epoch > 10 and accuracy[-1] > accuracy[-2]:
            epoch_max: int = epoch
            break

    print(f"Finished training. Best accuracy: {100 * accuracy[-2]:.2f}% in epoch {epoch_max}")
    print(accuracy)
    return model, accuracy


def save_model_structure(model: CNN, accuracy, save_weight: bool = False):
    """
    Saves the model structure to a file.

    Parameters:
        save_weight: bool
            If the model weights should be saved.
        model: nn.Module
            The neural network model.
        accuracy: float
            The accuracy of the model.
    """
    move_working_directory()

    with open('model_results.txt', 'a') as f:
        f.write(f'{100 * accuracy[-2]:.2f}% {str(model)}\n')

    if save_weight:
        filename = get_new_filename('ckpt')
        torch.save(model, filename)
        print(f"Model weights saved to {filename}")





def check_accuracy(loader, model, device, printing=False):

    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        printing: Choose whether to print the accuracy or not.
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
        accuracy = float(num_correct) / float(num_samples)
        if printing:
            if loader.dataset.train:
                print(f"train: Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
            else:
                print(f"test:  Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    model.train()  # Set the model back to training mode
    return accuracy