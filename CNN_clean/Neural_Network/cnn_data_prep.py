import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

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


def data_prep(data, logger, args):
    """
    Prepares the data for training and testing by creating DataLoader objects.
    Args:
        data (tuple): Tuple containing the training and testing data
        args (dict): Dictionary containing the arguments for DataLoader
    Returns:
        train_loader (DataLoader): DataLoader for the training data
        test_loader (DataLoader): DataLoader for the testing data
    """
    test_size = args.get('test_size', 0.3)
    event_ratio = args.get('swarm_event_ratio', 0.5)

    print(data)
    y_all_oh, y_all, x_all = data
    #---Mixing
    # Jetzt Split logik nach Verhältnissen (aber alles ist schon normiert + encoded)
    idx_swarm = np.where(y_all == 1)[0]
    idx_no_swarm = np.where(y_all == 0)[0]

    min_class_count = min(len(idx_swarm), len(idx_no_swarm))
    idx_swarm = idx_swarm[:min_class_count]
    idx_no_swarm = idx_no_swarm[:min_class_count]

    test_count_swarm = int(min_class_count * test_size * event_ratio)
    test_count_no = int(min_class_count * test_size * (1 - event_ratio))

    # Test/Train Indexwahl
    test_swarm = np.random.choice(idx_swarm, size=test_count_swarm, replace=False)
    train_swarm = np.setdiff1d(idx_swarm, test_swarm)

    test_no = np.random.choice(idx_no_swarm, size=test_count_no, replace=False)
    train_no = np.setdiff1d(idx_no_swarm, test_no)

    # Finales Shuffle & Auswahl
    train_idx = np.concatenate([train_swarm, train_no])
    test_idx = np.concatenate([test_swarm, test_no])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    x_train_norm = x_all[train_idx]
    y_train_transformed = y_all_oh[train_idx]

    x_test_norm = x_all[test_idx]
    y_test_transformed = y_all_oh[test_idx]
    y_test = y_all[test_idx]  # original labels für spätere Auswertung

    input_shape = x_train_norm.shape[1:]
    n_classes = y_train_transformed.shape[1]

    # ---------------------------------------------------------------
    logger.debug(f"Train set:   {x_train_norm.shape}, {y_train_transformed.shape}")
    logger.debug(f"Test set:    {x_test_norm.shape}, {y_test_transformed.shape}")
    logger.debug(f"Norm arrays: {x_train_norm.shape}, {x_test_norm.shape}")
    # ---------------------------------------------------------------

    # --------------------------------------------------------------
    logger.info(f"Ratio of test data: set value {test_size}; is value {len(x_test_norm) / len(x_train_norm)}")
    logger.info(
        f"Ratio of training data: set value {test_size}; is value {len(y_test_transformed) / len(y_train_transformed)}")
    logger.info(
        f"train_data is made of {np.sum(np.argmax(y_train_transformed, axis=1) == 1)} random swarm mels and {np.sum(np.argmax(y_train_transformed, axis=1) == 0)} random non swarm mels.")
    logger.info(
        f"test_data  is made of {np.sum(np.argmax(y_test_transformed, axis=1) == 1)} random swarm mels and {np.sum(np.argmax(y_test_transformed, axis=1) == 0)} random non swarm mels.")
    # ---------------------------------------------------------------

    try:
        assert n_classes == data[4]
    except AssertionError:
        logger.debug(f"Mismatch in class count: {n_classes} vs {data[4]}")

    #return input_shape, n_classes, x_train_norm, y_train_transformed, x_test_norm, y_test_transformed, y_test, data[3], \data[4]

    #---Mixing
    '''
    Initialize variables
    x_train_norm = data[2]
    y_train_transformed = data[3]
    x_test_norm = data[4]
    y_test_transformed = data[6]
    '''
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
