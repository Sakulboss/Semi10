import torch
from torch.utils.data import DataLoader, Dataset


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


def data_prep(data, args):

    x_train_norm = data[2]
    y_train_transformed = data[3]
    x_test_norm = data[4]
    y_test_transformed = data[6]

    x_train_tensor = torch.tensor(x_train_norm, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_transformed, dtype=torch.float)
    x_test_tensor = torch.tensor(x_test_norm, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_transformed, dtype=torch.float)

    train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
    test_dataset = CustomDataset(x_test_tensor, y_test_tensor, train=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args['batch_size'], shuffle=True)

    return train_loader, test_loader
