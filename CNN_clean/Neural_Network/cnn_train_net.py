import json
import time

import requests
import torch
from torch import optim, no_grad, nn
from tqdm import tqdm
import os
import numpy as np

from CNN_clean.Neural_Network.cnn_helpers import get_uuid
from CNN_clean.Neural_Network.cnn_net_prep import CNN
from cnn_net_prep import CNN


def move_working_directory():
    """
        This function changes the current working directory (cwd) to the directory 'modelle'. In this directory, the model is saved.
        Returns:
            None
        """

    os.chdir('..')
    os.chdir('files')


def get_new_filename(file_extension: str) -> str:
    """
    Creates a new filename for the model based on the number of existing files with the same extension in the current working directory.
    Args:
        file_extension: name of the file extension (e.g. 'ckpt', 'pt', ...)
    Returns:
        new filename
    """
    count = len([counter for counter in os.listdir(os.getcwd()) if counter.endswith('.'+file_extension)]) + 1
    return f'model_torch_{count}.{file_extension}'


def train(loader, args, logger) -> CNN | None:
    """
    Trains the model using the given data loader and arguments.
    Args:
        loader: data loader for the training and testing data
        args:   dictionary containing training parameters such as epochs, learning rate, if it should print the accuracy, etc.
        logger: logger for logging
    Returns:
        the trained model and the accuracy of the trained model
    """

    # Initialize variables
    train_loader    = loader[0]
    test_loader     = loader[1]
    max_epochs      = args.get('max_epochs', 10)
    learning_rate   = args.get('learning_rate', 0.01)
    min_epoch       = args.get('min_epoch', 5)
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #Create the model and check if the model is working -> when all models are tested, model.working is False. After that move model to GPU or CPU.
    model = CNN(logger, args)
    if model.working is False:
        return None
    model = model.to(device=device)

    #Initialize the loss function and optimizer as well as the timer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start = time.perf_counter()

    for epoch in range(max_epochs):
        logger.debug(f"Epoch [{epoch + 1}/{max_epochs}]")
        for batch_index, (data, targets) in enumerate(tqdm(train_loader, disable=True)):

            # Move data and targets to the device (GPU/CPU)
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = model(data)

            try:
                loss = criterion(scores, targets)
            except RuntimeError:
                logger.critical(f'Shape of scores ({scores.shape}) or targets ({targets.shape}) are not equal -> Training cannot be completed')
                logger.debug(f'Scores: {scores} \nTargets: {targets}')
                break

            # Backward pass: compute the gradients
            optimizer.zero_grad()
            loss.backward()

            # Optimization step: update the model parameters
            optimizer.step()

        # measure the time for each epoch, it will be later used to calculate the average time for each epoch
        end = time.perf_counter()
        model.epoch_time.append(end - start)
        start = end

        #calculate the middle squared error of the model, if it gets worse, stop training.

        acc = check_accuracy(test_loader, model, device, logger)
        model.accuracy.append((1-acc)**2)
        model.mse.append(sum(model.accuracy)/len(model.accuracy))
        #if epoch > min_epoch and model.mse[-1] > model.mse[-2]:
        if epoch > min_epoch and model.accuracy[-1] > model.accuracy[-2]:
            model.epoch_max = epoch
            break
    else:
        model.epoch_max = max_epochs
    logger.info(f"Finished training. MSE: {model.mse[-2]:.2f} in epoch {model.epoch_max} with on average {sum(model.epoch_time)/len(model.epoch_time):.3f} s and model {str(model)}")
    return model


def save_model_structure(model: CNN, logger, args):
    """
    Saves the model structure to a file.
    Parameters:
        args:  Settings for the model, including the path to save the model.
        logger: The logger for logging.
        model: The neural network model.
    Returns:
        None
    """

    path = args.get('dropbox', None)
    save_weight = args.get('save_weight', False)

    if path is None:
        move_working_directory()
        path = os.getcwd()
    path_to_file = os.path.join(path, 'model_results.txt')

    with open(path_to_file, 'a') as f:
        f.write(f'{100 * model.accuracy[-2]:.5f}% {str(model)}\n')

    send_result(model.accuracy[-2], args, logger)

    if save_weight:
        os.chdir(path)
        filename = get_new_filename('ckpt')
        torch.save(model, filename)
        logger.debug(f"Model weights saved to {filename}")


def send_result(model, args, logger):
    """
    This function sends the result of the training back to the server.
    Args:
        model:         The trained model.
        args:        The arguments for the training.
        logger:      The logger for logging.
    Returns:
        None
    """

    device_uuid = get_uuid(args.get('device_uuid', 'device_uuid'))
    server_url = args.get('server_url', 'https://survive.cermann.com/server.php')

    headers = {'Content-Type': 'application/json'}
    payload = {'line_index': int(model),
               'model':      str(model),
               'result':     model.__acc__(),
               'epoch':      model.__epoch__()}
    params = {'key': device_uuid}
    logger.debug(f"Sending result to server: {payload}")

    try:
        logger.debug(f"POST {server_url} with payload {json.dumps(payload)}")
        response = requests.post(server_url, params=params, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.debug('Server response:', data.get('message', 'No response message'))
    except requests.RequestException as e:
        logger.critical('Error during POST:', str(e))
    except json.JSONDecodeError:
        logger.critical('Error decoding JSON response from server')


def check_accuracy(loader, model, device, logger):
    """
    Checks the accuracy of the model on the given dataset loader.
    Parameters:
        loader: DataLoader The DataLoader for the dataset to check accuracy on.
        device: string The Device to run the model on.
        model: nn.Module The neural network model.
        logger: logger The logger for logging.
    """

    # Initialize variables
    num_correct = 0
    num_samples = 0

    # Set the model to evaluation mode -> disables dropout and batch normalization
    model.eval()

    # Disable gradient calculation -> should only test model performance
    with no_grad():
        for x, y in loader:

            x = x.to(device)
            _, predictions = model(x).max(1)
            predictions_new = np.array(predictions.cpu())

            # Don't know why this is needed, but without it the calculation doesn't work
            if loader.dataset.train:
                y_new = np.array(y.max(1))
                y_new = np.array([int(i) for i in y_new[1]])
            else:
                y_new = np.array([int(i) for i in y])

            # Calculate the number of correct predictions (takes the sum of the correct predictions pairs (if it works, it works))
            num_correct += (predictions_new == y_new).sum()
            num_samples += predictions.size(0)

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples)

        if loader.dataset.train:
            logger.debug(f"train: Got {num_correct}/{num_samples} with accuracy {100*accuracy:.2f}%")
        else:
            logger.debug(f"test:  Got {num_correct}/{num_samples} with accuracy {100*accuracy:.2f}%")

    model.train()  # Set the model back to training mode
    return accuracy
