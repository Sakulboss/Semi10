import torch
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


def data_prep(data, args):
    """
    Prepares the data for training and testing by creating DataLoader objects.
    Args:
        data (tuple): Tuple containing the training and testing data
        args (dict): Dictionary containing the arguments for DataLoader
    Returns:
        train_loader (DataLoader): DataLoader for the training data
        test_loader (DataLoader): DataLoader for the testing data
    """

    # Initialize variables
    x_train_norm = data[2]
    y_train_transformed = data[3]
    x_test_norm = data[4]
    y_test_transformed = data[6]

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
