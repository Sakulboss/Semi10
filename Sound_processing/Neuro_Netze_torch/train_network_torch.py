import torch
from torch import optim
from torch import nn
from tqdm import tqdm
from Sound_processing.Neuro_Netze_torch.network_prep import CNN, check_accuracy
import os

def move_working_directory():
    working_directory = os.getcwd()
    for i in range(3):
        if os.path.basename(working_directory) != "Sound_processing":
            os.chdir('..')
            break
    os.chdir('Neuro_Netze_torch')

def get_new_filename(file_extension: str) -> str:
    count = len([counter for counter in os.listdir('C:\\modelle') if counter.endswith(file_extension)]) + 1
    return f'model_torch_{count}.{file_extension}'

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
    move_working_directory()
    print(os.getcwd())
    torch.save(model, get_new_filename('ckpt'))