import json
import time

import requests
import torch
from torch import optim, no_grad, nn
from tqdm import tqdm
import os
import numpy as np
from cnn_net_prep import CNN


def move_working_directory():
    """
        This function changes the current working directory (cwd) to the directory 'modelle'. In this directory, the model is saved.
        Returns:
            None
        """
    working_directory = os.getcwd()
    for i in range(3):
        if os.path.basename(working_directory) != "Sound_processing":
            os.chdir('../..')
            break
    os.chdir('modelle')


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


def train(loader, args, logger) -> tuple[CNN, list] | None:
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
    accuracy        = []
    epoch_time      = []
    mse             = []

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
        epoch_time.append(end - start)
        start = end

        #calculate the middle squared error of the model, if it gets worse, stop training.

        acc = check_accuracy(test_loader, model, device, logger)
        accuracy.append((1-acc)**2)
        mse.append(sum(accuracy)/len(accuracy))
        if epoch > min_epoch and mse[-1] > mse[-2]:
            epoch_max: int = epoch
            break
    else:
        epoch_max: int = max_epochs
    logger.info(f"Finished training. MSE: {100*mse[-1]:.2f}% in epoch {epoch_max} with on average {sum(epoch_time)/len(epoch_time):.3f} s and model {str(model)}")
    return model, accuracy


def save_model_structure(model: CNN, accuracy, path = None, save_weight: bool = False):
    """
    Saves the model structure to a file.

    Parameters:
        save_weight: bool
            If the model weights should be saved.
        model: nn.Module
            The neural network model.
        accuracy: list
            The accuracy of the model.
        path: str
            The path to the dropbox to save the model structure. If None, it will be saved in models.
    Returns:
        None
    """


    if path is None:
        move_working_directory()
        path = os.getcwd()
    path_to_file = os.path.join(path, 'model_results.txt')

    with open(path_to_file, 'a') as f:
        f.write(f'{100 * accuracy[-2]:.5f}% {str(model)}\n')

    if save_weight:
        os.chdir(path)
        filename = get_new_filename('ckpt')
        torch.save(model, filename)
        print(f"Model weights saved to {filename}")


def send_result(device_uuid, line_index, result, server_url, logger):
    """
    This function sends the result of the training back to the server.
    Args:
        device_uuid: The UUID of the device.
        line_index:  The index of the line that was trained.
        result:      The result of the training.
        logger:      The logger for logging.
    Returns:
        None
    """

    headers = {'Content-Type': 'application/json'}
    payload = {'line_index': line_index, 'result': result}
    params = {'key': device_uuid}

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
        device: string The Device to run the model on.
        loader: DataLoader The DataLoader for the dataset to check accuracy on.
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
            logger.debug(f"train: Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
        else:
            logger.debug(f"test:  Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")

    model.train()  # Set the model back to training mode
    return accuracy
