from typing import Any

import matplotlib.pyplot as pl
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalMaxPooling2D, BatchNormalization
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import sys

printing: bool = False
file_ = sys.stdout

def printer(*text, sep=' ', end='\n', file=None):
    global printing
    global file_
    file = file or file_
    printer(*text, sep=sep, end=end, file=file)

def get_new_filename(file_extension: str) -> str:
    count = len([counter for counter in os.listdir('C:\\modelle') if counter.endswith(file_extension)]) + 1
    return f'full_model_{count}.{file_extension}'

def model_creation(input_shape, n_classes):
    # Define the model
    model = Sequential()
    model.add(Input(shape=input_shape))
    # 1st Convolutional Layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    # Flatten the output to feed into the Dense layer
    model.add(Flatten())

    # Output layers for classification
    model.add(Dense(units=64, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))
    model.add(Dense(units=n_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def model_training(data: tuple, setting: dict) -> tuple[Any, Any, Any]:
    global printing, file_
    printing = setting.get('print', False)
    file_ = setting.get('file', sys.stdout)

    input_shape = setting.get('input_shape', data[0])
    n_classes = setting.get('n_classes', data[1])
    X_train_norm = setting.get('X_train_norm', data[2])
    y_train_transformed = setting.get('y_train_transformed', data[3])
    epochs = setting.get('epochs', 1)
    batch_size = setting.get('batch_size', 4)


    model = setting.get('model', model_creation(input_shape, n_classes))

    history = model.fit(X_train_norm,
                        y_train_transformed,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2)

    if setting.get('plot_history', False):
        pl.figure(figsize=(8, 4))
        pl.subplot(2, 1, 1)
        pl.plot(history.history['loss'])
        pl.ylabel('Loss')
        pl.subplot(2, 1, 2)
        pl.plot(history.history['accuracy'])
        pl.ylabel('Accuracy (Training Set)')
        pl.xlabel('Epoch')
        pl.show()

    model.save(f'C:\\modelle\\{get_new_filename("keras")}')  # saves the model into modelle
    return model, history, data


def model_evaluation(data, setting):
    global printing, file_
    printing = setting.get('print', False)
    file_ = setting.get('file', sys.stdout)

    trained_model = setting.get('trained_model', None)
    X_test_norm = setting.get('X_test_norm', data[4])
    y_test = setting.get('y_test', data[6])
    unique_classes = setting.get('unique_classes', data[7])
    n_sub = setting.get('n_sub', data[8])
    printing = setting.get('printinge', False)

    if trained_model is None:
        model = model_training(**setting)[0]
    else: model = trained_model

    printer("Shape of the test data: {}".format(X_test_norm.shape))
    y_test_pred = model.predict(X_test_norm)
    printer("Shape of the predictions: {}".format(y_test_pred.shape))

    # The model outputs in each row 5 probability values (they always add to 1!) for each class.
    # We want to take the class with the highest probability as prediction!

    y_test_pred = np.argmax(y_test_pred, axis=1)
    printer(y_test_pred)
    printer("Shape of the predictions now: {}".format(y_test_pred.shape))

    # Accuracy
    accuracy = accuracy_score(y_test, y_test_pred)
    printer("Accuracy score = ", accuracy)

    if setting.get('confusion_matrix', False):
        pl.figure(figsize=(50, 50))
        cm = confusion_matrix(y_test, y_test_pred).astype(np.float32)
        # normalize to probabilities
        for i in range(cm.shape[0]):
            if np.sum(cm[i, :]) > 0:
                cm[i, :] /= np.sum(cm[i, :])  # by dividing through the sum, we convert counts to probabilities
        pl.imshow(cm)
        ticks = np.arange(n_sub)
        pl.xticks(ticks, unique_classes)
        pl.yticks(ticks, unique_classes)
        pl.show()


    return unique_classes