import torch
from torch import optim
from torch import nn
from tqdm import tqdm
from Sound_processing.Neuro_Netze_torch.network_prep import CNN, check_accuracy



def train(loader, args):

    train_loader = loader[0]
    test_loader = loader[1]
    num_epochs = args.get('epochs', 10)
    num_classes = args.get('num_classes')
    learning_rate = args.get('learning_rate')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        check_accuracy(test_loader, model, device)

    check_accuracy(train_loader, model, device)