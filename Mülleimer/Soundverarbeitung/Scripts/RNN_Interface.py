import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import SimpleRNN, Dense
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

def rnnTraining(text: str, seq_length = 3):
    chars = sorted(list(set(text)))
    char_to_index = {char: i for i, char in enumerate(chars)}
    index_to_char = {i: char for i, char in enumerate(chars)}

    sequences = []
    labels = []

    for i in range(len(text) - seq_length):
        seq = text[i:i + seq_length]
        label = text[i + seq_length]
        sequences.append([char_to_index[char] for char in seq])
        labels.append(char_to_index[label])

    X = np.array(sequences)
    y = np.array(labels)

    X_one_hot = tf.one_hot(X, len(chars))
    y_one_hot = tf.one_hot(y, len(chars))

    model = Sequential()
    model.add(SimpleRNN(100, input_shape=(seq_length, len(chars)), activation='relu'))
    model.add(Dense(len(chars), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_one_hot, y_one_hot, epochs=500)
    return model, char_to_index, index_to_char, seq_length

def textgenerierung(start_seq: str, model) -> str:
    generated_text = start_seq

    for i in range(45):
        x = np.array([[char_to_index[char] for char in generated_text[-seq_length:]]])
        x_one_hot = tf.one_hot(x, len(chars))
        prediction = model.predict(x_one_hot)
        next_index = np.argmax(prediction)
        next_char = index_to_char[next_index]
        generated_text += next_char

    return generated_text

def textausgabe(generated) -> None:
    print("Generated Text:")
    print(generated)
    print('\nThe length of the generated text is: ', len(generated))

