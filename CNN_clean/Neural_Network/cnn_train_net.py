import json
import time
import requests
import torch
from torch import optim, no_grad, nn
from tqdm import tqdm
import os
import logging

from cnn_helpers import get_uuid
from cnn_net_prep import CNN


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


def move_working_directory(target='files') -> None:
    """
    This function changes the current working directory (cwd) to the specified target directory.
    Args
        target: str The target directory to change to, default is 'files'.
    Returns:
        None
    """
    os.chdir('..')
    os.chdir(target)


def get_new_filename(file_extension: str, text:str = 'model_torch_') -> str:
    """
    Creates a new filename for the model based on the number of existing files with the same extension in the current working directory.
    Args:
        file_extension: str name of the file extension (e.g. 'ckpt', 'pt', ...)
        text:           str prefix for the filename, default is 'model_torch_'.
    Returns:
        New filename that doesn't exist in the directory.
    """
    count = len([counter for counter in os.listdir(os.getcwd()) if counter.endswith('.'+file_extension)]) + 1
    return f'{text}{count}.{file_extension}'


def train(loader,  logging_args, args) -> tuple[CNN, float, int] | tuple[None, None, None]:
    """
    Trains the model using the given data loader and arguments.
    Args:
        loader:       DataLoaderdata  loader for the training and testing data
        logging_args: dict            arguments for logging
        args:         dict            dictionary containing training parameters such as epochs, learning rate, etc.
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
    logger          = setup_logging(logging_args)

    # Create the model and check if the model is working -> when all models are tested, model.working is set False, it should then break the training.
    model = CNN(logger, args)
    if model.working is False:
        return None, None, None

    # Move the model to the GPU if available.
    model = model.to(device=device)

    #Initialize the loss function and optimizer as well as the timer for epoch timing.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start = time.perf_counter()

    # Start training loop
    for epoch in range(max_epochs):
        logger.debug(f"Epoch [{epoch + 1}/{max_epochs}]")
        # Iterate over the batch
        for batch_index, (data, targets) in enumerate(tqdm(train_loader, disable=True)):

            # Move data and targets to the GPU if available.
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass: compute the model output
            scores = model(data)

            # This was used for debugging, but it's safe to keep for just in case.
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

        # Measure the time for each epoch, it will be later used to calculate the average time for each epoch.
        end = time.perf_counter()
        model.epoch_time.append(end - start)
        start = end

        # calculate the accuracy (Richtigkeit) of the model on the test data set, on which the model was not trained.
        acc = check_accuracy(test_loader, model, device, logger)

        # Some models have the accuracy set to 0.5, which means probably too many neurons are dead, so that only one class is predicted every time, which means that 50% are correct of the test data because it is split equal into the events..
        if acc == 0.5:
            return model, acc, -1

        # calculate the squared error and check if it gets bigger (worse), then break the training loop.
        model.se_func((1-acc)**2)
        if epoch > min_epoch and model.se[-1] > model.se[-2]:
            # Set the highest reached epoch.
            model.epoch(epoch)
            break
    else:
        model.epoch(max_epochs)

    logger.info(f"Finished training with SE: {model.se[-2]:.2f} in epoch {model.epoch()} in on average {sum(model.epoch_time)/len(model.epoch_time):.3f} s and model {str(model)}")
    return model, model.acc()[-2], model.epoch()


def save_model_structure(model: CNN, acc: float, epoch: int, logging_args: dict, args: dict) -> None:
    """
    Saves the model structure to a file.
    Parameters:
        model:        CNN   The neural network model.
        acc:          float accuracy of the model
        epoch:        int   max epoch number reached in training
        logging_args: dict  arguments for logging
        args:         dict  settings for the training
    Returns:
        None
    """
    logger = setup_logging(logging_args)
    path = args.get('dropbox', None)
    save_weight = args.get('save_weight', False)

    if path is None:
        move_working_directory()
        path = os.getcwd()
    path_to_file = os.path.join(path, 'model_results.txt')

    # Even if use_server is activated, the results are saved to a file in case the sending goes wrong
    with open(path_to_file, 'a') as f:
        f.write(f'{100 * model.accuracy[-2]:.5f}% {str(model)}\n')

    if args.get('use_server',  False): send_result(model.accuracy[-2], acc, epoch, args, logger)

    if save_weight:
        os.chdir(path)
        filename = get_new_filename('ckpt')
        torch.save(model, filename)
        logger.debug(f"Model weights saved to {filename}")


def send_result(model, acc, epoch, logging_args, args):
    """
    This function sends the result of the training back to the server.
    Args:
        model:        CNN   trained model
        acc:          float accuracy of the model
        epoch:        int   highest epoch reached in training epoch
        args:         dict  arguments for training, also containing the ones for server interaction
        logging_args: dict  arguments for logging
    Returns:
        None
    """
    # Initilize variables
    logger = setup_logging(logging_args)
    device_uuid = get_uuid(args.get('device_uuid', 'uuid.txt'))
    server_url = args.get('server_url', 'https://survive.cermann.com/server.php')
    headers = {'Content-Type': 'application/json'}
    payload = {'line_index': int(model),
               'model':      str(model),
               'result':     acc,
               'epoch':      epoch}
    params = {'key': device_uuid}

    # Try accessing the server and sending the result, if it fails, log the error.
    logger.debug(f"Sending result to server: {payload}")
    try:
        logger.debug(f"POST {server_url} with payload {json.dumps(payload)}")
        response = requests.post(server_url, params=params, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        logger.debug('Server response:', data.get('message', 'No response message'))
    except requests.RequestException as e:
        logger.critical(f'Error during POST: {str(e)}')
    except json.JSONDecodeError:
        logger.critical('Error decoding JSON response from server')


def check_accuracy(loader, model, device, logger) -> float:
    """
    Checks the accuracy of the model on the given dataset loader.
    Parameters:
        loader: DataLoader  The DataLoader for the dataset to check accuracy on.
        device: string The  Device to run the model on.
        model:  nn.Module   The neural network model.
        logger: logger      The logger for logging.
    Returns:
        accuracy: float       The accuracy of the Epoch
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
            y = y.to(device)

            # Get model predictions
            scores = model(x)
            _, predictions = scores.max(1)  # shape: [batch_size]

            # Convert one-hot encoded labels to class indices if necessary;
            # Is necessary, because we OneHotEncode first -> this needs OneHotDecoded
            if y.ndim == 2 and y.size(1) > 1:
                y = y.argmax(dim=1)

            # Compare predictions to labels
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples)

        if loader.dataset.train:
            logger.debug(f"train: Got {num_correct}/{num_samples} with accuracy {100*accuracy:.2f}%")
        else:
            logger.debug(f"test:  Got {num_correct}/{num_samples} with accuracy {100*accuracy:.2f}%")

    # Set the model back to training mode
    model.train()
    return accuracy
