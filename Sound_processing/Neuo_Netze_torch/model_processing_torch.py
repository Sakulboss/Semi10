import matplotlib.pyplot as pl
#from tensorflow.keras.models import Sequential, load_model, Model
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalMaxPooling2D, BatchNormalization
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import haupt_driver_torch as hdt

printing: bool = False
file_ = sys.stdout

class CNNModel(nn.Module):
    def __init__(self, input_shape, n_classes):
        super(CNNModel, self).__init__()

        # Definiere die Schichten des Modells
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Berechne die Größe nach den Convolutional und Pooling Schichten
        # Hier nehmen wir an, dass die Eingabe ein 4D Tensor ist: (batch_size, channels, height, width)
        self.flatten_size = self._get_flatten_size(input_shape)

        self.fc1 = nn.Linear(self.flatten_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)

    def _get_flatten_size(self, input_shape):
        # Dummy input to calculate the size after convolutions and pooling
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)  # Batch size 1
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.pool(x)
            return x.numel()  # Anzahl der Elemente im Tensor

    def forward(self, x):
        # Definiere den Vorwärtsdurchlauf
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def printer(*text, sep=' ', end='\n', file=None):
    global printing
    global file_
    file = file or file_
    if printing: print(*text, sep=sep, end=end, file=file)


def get_new_filename(file_extension: str) -> str:
    count = len([counter for counter in os.listdir('C:\\modelle') if counter.endswith(file_extension)]) + 1
    return f'full_model_{count}.{file_extension}'


def model_training_torch(data, setting):
    global printing, file_
    printing = setting.get('print', False)
    file_ = setting.get('file', sys.stdout)

    input_shape = setting.get('input_shape', data[0])
    n_classes = setting.get('n_classes', data[1])
    X_train_norm = setting.get('X_train_norm', data[2])
    y_train_transformed = setting.get('y_train_transformed', data[3])
    epochs = setting.get('epochs', 1)
    batch_size = setting.get('batch_size', 4)
    model = setting.get('model', CNNModel(input_shape, n_classes))
    device = setting.get('device', 'cuda') # types: cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia
    model.train().to(device)

    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()
    history = {'loss': [], 'accuracy': []}

    # Konvertiere die Daten in PyTorch Tensoren
    X_train_tensor = torch.tensor(X_train_norm, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_transformed, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(X_train_tensor), batch_size):
            # Batch erstellen
            inputs = X_train_tensor[i:i + batch_size]
            labels = y_train_tensor[i:i + batch_size]

            # Gradienten zurücksetzen
            optimizer.zero_grad()

            # Vorwärtsdurchlauf
            outputs = model(inputs)
            output = loss(outputs, labels)

            # Rückwärtsdurchlauf und Optimierung
            output.backward()
            optimizer.step()

            # Verlust und Genauigkeit berechnen
            running_loss += output.item()
            #_, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (outputs == labels).sum().item()

        # Durchschnittlichen Verlust und Genauigkeit für die Epoche speichern
        epoch_loss = running_loss / (len(X_train_tensor) // batch_size)
        epoch_accuracy = correct / total
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)

        if printing:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}', file=file_)

    # Plot der Trainingshistorie
    if setting.get('plot_history', False):
        pl.figure(figsize=(8, 4))
        pl.subplot(2, 1, 1)
        pl.plot(history['loss'])
        pl.ylabel('Loss')
        pl.subplot(2, 1, 2)
        pl.plot(history['accuracy'])
        pl.ylabel('Accuracy (Training Set)')
        pl.xlabel('Epoch')
        pl.show()

    # Modell speichern
    torch.save(model.state_dict(), f'C:\\modelle\\{get_new_filename("pytorch")}.pth')  # Speichert das Modell

    return model, history, data

if __name__ == '__main__':
    hdt.main(settings={'print': True, 'big' : True, 'file': file_, 'confusion_matrix' : True, 'epochs' : 20000, 'batch_size' : 128})
    #pass


"""
Bei diesem Programm tritt folgender Fehler auf:
"""