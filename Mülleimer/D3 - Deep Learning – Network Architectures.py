import glob
import os
import numpy as np
import matplotlib.pyplot as pl
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import Sound_processing.training_files.datensatz as msc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

#Herunterladen der Sounddateien und entpacken --------------------------------------------------------------------------

msc.download_small_dataset() # für den kleinen Datensatz
#msc.big_dataset()
dir_dataset = '../Sound_processing/_animal_sounds'
#dir_dataset = '_viele_sounds_geordnet'
sub_directories = glob.glob(os.path.join(dir_dataset, '*'))

#Classification --------------------------------------------------------------------------------------------------------


n_sub = len(sub_directories)
# let's collect the files in each subdirectory
# the folder name is the class name
fn_wav_list = []
class_label = []
file_num_in_class = []

for i in range(n_sub):
    current_class_label = os.path.basename(sub_directories[i])
    current_fn_wav_list = sorted(glob.glob(os.path.join(sub_directories[i], '*.wav')))
    for k, fn_wav in enumerate(current_fn_wav_list):
        fn_wav_list.append(fn_wav)
        class_label.append(current_class_label)
        file_num_in_class.append(k)

n_files = len(class_label)

# this vector includes a "counter" for each file within its class, we use it later ...
file_num_in_class = np.array(file_num_in_class)

unique_classes = sorted(list(set(class_label)))
print("All unique class labels (sorted alphabetically): ", unique_classes)
class_id = np.array([unique_classes.index(_) for _ in class_label])

#mel_specs erstellen ---------------------------------------------------------------------------------------------------


all_mel_specs = []
for i, fn_wav in enumerate(fn_wav_list):
    all_mel_specs.append(msc.compute_mel_spec_for_audio_file(fn_wav_list[i]))

print("We have {} spectrograms of shape: {}".format(len(all_mel_specs), all_mel_specs[0].shape))

all_mel_specs = np.array(all_mel_specs)
print(f"Shape of our data tensor:         {all_mel_specs.shape}")

segment_list = []
segment_file_id = []
segment_class_id = []
segment_spec_id = []

n_spectrograms = all_mel_specs.shape[0]

n_segments_per_spectrogram = 10
segment_length_frames = 100

spec_length_frames = all_mel_specs.shape[2]
max_segment_start_offset = spec_length_frames - segment_length_frames

# iterate over all spectrograms
for i in range(n_spectrograms):

    # create [n_segments_per_spectrogram] segments
    for s in range(n_segments_per_spectrogram):
        # random segment start frame
        segment_start_frames = int(np.random.rand(1).item() * max_segment_start_offset)

        segment_list.append(all_mel_specs[i, :, segment_start_frames:segment_start_frames + segment_length_frames])

        segment_file_id.append(i)
        segment_class_id.append(class_id[i])
        segment_spec_id.append(s)

# finally, let's convert our list of spectrogram segments again into a 3D tensor
segment_list = np.array(segment_list)

segment_file_id = np.array(segment_file_id)
segment_file_mod_id = np.mod(segment_file_id, 5)

segment_class_id = np.array(segment_class_id)
segment_spec_id = np.array(segment_spec_id)

print(f"New data tensor shape:            {segment_list.shape}")

pl.figure(figsize=(12, 4))
pl.plot(segment_file_id, 'b-', label='segment file ID')
pl.plot(segment_file_mod_id, 'b--', label='segment file ID (per spectrogram)')
pl.plot(segment_class_id, label='segment class ID')
pl.plot(segment_spec_id, label='segment ID')
pl.legend()
pl.xlabel('Segment')
pl.show()

pl.figure(figsize=(2.5, 2))
pl.imshow(all_mel_specs[0, :, :], origin="lower", aspect="auto", interpolation="None")
pl.xticks([], [])
pl.yticks([], [])
pl.title('Original spectrogram')
pl.tight_layout()
pl.show()

pl.figure(figsize=(15, 5))
ny = 2
nx = int(n_segments_per_spectrogram // ny)
for s in range(n_segments_per_spectrogram):
    pl.subplot(ny, nx, s + 1)
    pl.imshow(segment_list[s, :, :], origin="lower", aspect="auto", interpolation="None")
    if s == 0:
        pl.title('Extracted segments')
    pl.xticks([], [])
    pl.yticks([], [])
pl.tight_layout()
pl.show()

is_train = np.where(segment_file_mod_id <= 2)[0]
is_test = np.where(segment_file_mod_id >= 3)[0]

print("Our feature matrix is split into {} training examples and {} test examples".format(len(is_train), len(is_test)))

X_train = segment_list[is_train, :, :]
y_train = segment_class_id[is_train]
X_test = segment_list[is_test, :, :]
y_test = segment_class_id[is_test]

print("Let's look at the dimensions")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train_norm = np.zeros_like(X_train)
X_test_norm = np.zeros_like(X_test)

for i in range(X_train.shape[0]):
    X_train_norm[i, :, :] = StandardScaler().fit_transform(X_train[i, :, :])

for i in range(X_test.shape[0]):
    X_test_norm[i, :, :] = StandardScaler().fit_transform(X_test[i, :, :])

y_train_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_train.reshape(-1, 1))
y_test_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_test.reshape(-1, 1))

if len(X_train_norm.shape) == 3:
    X_train_norm = np.expand_dims(X_train_norm, -1)
    X_test_norm = np.expand_dims(X_test_norm, -1)

else:
    print("We already have four dimensions")

print(f"Let's check if we have four dimensions. New shapes: {X_train_norm.shape} & {X_test_norm.shape}")

# The input shape is the "time-frequency shape" of our segments + the number of channels
# Make sure to NOT include the first (batch) dimension!
input_shape = X_train_norm.shape[1:]

# Get the number of classes:
n_classes = y_train_transformed.shape[1]

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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

checkpoint_path = "training_1/.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

checkpoint_path = "C:\\modelle"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=2)]

history = model.fit(X_train_norm,
                    y_train_transformed,
                    epochs=1,
                    batch_size=4,
                    verbose=2,
                    callbacks=cp_callback)

pl.figure(figsize=(8, 4))
pl.subplot(2, 1, 1)
pl.plot(history.history['loss'])
pl.ylabel('Loss')
pl.subplot(2, 1, 2)
pl.plot(history.history['accuracy'])
pl.ylabel('Accuracy (Training Set)')
pl.xlabel('Epoch')
pl.show()

y_test_pred = model.predict(X_test_norm)

print("Shape of the predictions: {}".format(y_test_pred.shape))

# The model outputs in each row 5 probability values (they always add to 1!) for each class.
# We want to take the class with the highest probability as prediction!

y_test_pred = np.argmax(y_test_pred, axis=1)
print("Shape of the predictions now: {}".format(y_test_pred.shape))

# Accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy score = ", accuracy)

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

model.save(f'C:\\modelle\\{msc.get_new_filename("keras")}') #saves the model into modelle

print("Done :)")