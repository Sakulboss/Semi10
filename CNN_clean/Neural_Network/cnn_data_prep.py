import torch
import numpy as np
import logging
from torch.utils.data import DataLoader, Dataset


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



class CustomDataset(Dataset):
    """Custom dataset class for loading features and labels. Workaround for the DataLoader to work with PyTorch.
    Args:
        features (torch.Tensor): Tensor containing the features
        labels (torch.Tensor): Tensor containing the labels
        train (bool): Flag indicating if the dataset is for training or testing
    """

    def __init__(self, features, labels, train = True):
        super(CustomDataset, self).__init__()
        self.features = features
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def data_prep(data, logging_args, args):
    """
    Prepares the data for training and testing by creating DataLoader objects.
    Args:
        data (tuple): Tuple containing the training and testing data
        logging_args (dict): Dictionary containing logging arguments
        args (dict): Dictionary containing the arguments for DataLoader
    Returns:
        train_loader (DataLoader): DataLoader for the training data
        test_loader (DataLoader): DataLoader for the testing data
    """

    logger = setup_logging(logging_args)

    # Get Args
    test_size = args.get('test_size', 0.3)
    event_ratio = args.get('swarm_event_ratio', 0.5)

    # Unpack data
    y_all_oh, y_all, x_all = data

    # Get indices of swarm and no swarm
    idx_swarm = np.where(y_all == 1)[0]
    idx_no_swarm = np.where(y_all == 0)[0]

    # Determine, how big classes can be (e.g. 500 swarm & 600 no_swarm -> only 500 of each will be chosen)
    min_class_count = min(len(idx_swarm), len(idx_no_swarm))

    # Get the whole class size, used for checking ratios
    whole_class_count = min_class_count * 2

    # Balance data_sets
    idx_swarm = idx_swarm[:min_class_count]
    idx_no_swarm = idx_no_swarm[:min_class_count]

    # Determine the size of each individual dataset
    test_count_swarm = int(int(test_size * whole_class_count) * event_ratio)
    test_count_no = int(test_size * whole_class_count) - test_count_swarm

    # Random choice from indices so nets don't learn it by hard; replace = False -> without laying back
    test_swarm = np.random.choice(idx_swarm, size=test_count_swarm, replace=False)
    train_swarm = np.setdiff1d(idx_swarm, test_swarm)

    test_no = np.random.choice(idx_no_swarm, size=test_count_no, replace=False)
    train_no = np.setdiff1d(idx_no_swarm, test_no)

    # Indices for train and test are put together
    train_idx = np.concatenate([train_swarm, train_no])
    test_idx = np.concatenate([test_swarm, test_no])

    # Shuffle indices for more "randomness"
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    # Finally "compile" train & test data
    # Gets training features (learn patterns)
    x_train_norm = x_all[train_idx]
    # Gets corresponding OneHotEncoded Labels
    y_train_transformed = y_all_oh[train_idx]

    # Gets testing features (learn patterns)
    x_test_norm = x_all[test_idx]
    # Gets corresponding OneHotEncoded Labels
    y_test_transformed = y_all_oh[test_idx]

    # -------------------------LOGGING TO CHECK EVERYTHING IS OKAY----------------------------------
    logger.info("---------------DATA---------------".center(75))
    logger.info(f"Train set:                                    {x_train_norm.shape}, {y_train_transformed.shape}")
    logger.info(f"Test set:                                     {x_test_norm.shape}, {y_test_transformed.shape}")
    logger.info(f"Norm arrays:                                  {x_train_norm.shape}, {x_test_norm.shape}")

    logger.info(f"Train data contains:                          {np.sum(np.argmax(y_train_transformed, axis=1) == 1)} swarm samples and {np.sum(np.argmax(y_train_transformed, axis=1) == 0)} non swarm samples.")
    logger.info(f"Test data contains:                           {np.sum(np.argmax(y_test_transformed, axis=1) == 1)} swarm samples and {np.sum(np.argmax(y_test_transformed, axis=1) == 0)} non swarm samples.")

    logger.info(f"Ratio of test data:                           set value {test_size}; is value {len(x_test_norm) / whole_class_count}")
    logger.info(f"Ratio of training data:                       set value {1 - test_size}; is value {len(x_train_norm) / whole_class_count}")

    logger.info(f"Ratio of swarm and non swarm in test data:    set value {event_ratio} / {1 - event_ratio}; is value {test_count_swarm / (test_count_swarm + test_count_no)} / {1 - test_count_swarm / (test_count_swarm + test_count_no)}")
    logger.info("---------------DATA---------------".center(75))
    # ----------------------------------------------------------------------------------------------

    # Create torch.tensors from the numpy arrays for easier computation and GPU support
    x_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_transformed, dtype=torch.float)
    x_test_tensor = torch.tensor(x_test_norm, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_transformed, dtype=torch.float)

    # Create CustomDataset objects for training and testing data
    train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
    test_dataset = CustomDataset(x_test_tensor, y_test_tensor, train=False)

    # Create DataLoader objects for training and testing data
    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], shuffle=True)

    return train_loader, test_loader
