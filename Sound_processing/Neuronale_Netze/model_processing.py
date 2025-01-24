import matplotlib.pyplot as pl
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalMaxPooling2D, BatchNormalization


def get_new_filename(file_extension: str) -> str:
    count = len([counter for counter in os.listdir('.\\modelle') if counter.endswith(file_extension)]) + 1
    return f'full_model_{count}.{file_extension}'

def model_creation(input_shape, n_classes, printing=False):
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

    if printing: model.summary()

    return model

def model_training(data, **kwargs):
    input_shape = kwargs.get('input_shape', data[0])
    n_classes = kwargs.get('n_classes', data[1])
    X_train_norm = kwargs.get('X_train_norm', data[2])
    y_train_transformed = kwargs.get('y_train_transformed', data[3])
    epochs = kwargs.get('epochs', 1)
    batch_size = kwargs.get('batch_size', 4)
    printing = kwargs.get('printing', False)

    model = kwargs.get('model', model_creation(input_shape, n_classes, printing=printing))

    history = model.fit(X_train_norm,
                        y_train_transformed,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2)

    if kwargs.get('plot_history', False):
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