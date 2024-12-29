import glob
import os
import librosa
import matplotlib.pyplot as pl
import numpy as np
from numpy import ndarray
from sklearn.metrics import accuracy_score, confusion_matrix
import IPython.display as ipd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
import mel_spec_calculator as msc


#Welche Diagramme angezeigt werden sollen ------------------------------------------------------------------------------


mel_spec_diagram: bool = False
mel_spec_diagram_inverse: bool = False
file_id_diagram: bool = False
accuracy_diagram: bool = False
confusion_matrix_diagram: bool = True
specto_predictions_diagram: bool = True


#Herunterladen der Sounddateien und entpacken --------------------------------------------------------------------------

msc.download_dataset()
dir_dataset = 'animal_sounds'
sub_directories = glob.glob(os.path.join(dir_dataset, '*'))


#Classification --------------------------------------------------------------------------------------------------------


n_sub = len(sub_directories)
# let's collect the files in each subdirectory
# the folder name is the class name
fn_wav_list: list|ndarray = []
class_label: list|ndarray = []
file_num_in_class: list|ndarray = []

for i in range(n_sub):
    current_class_label = os.path.basename(sub_directories[i])
    current_fn_wav_list = sorted(glob.glob(os.path.join(sub_directories[i], '*.wav')))
    for k, fn_wav in enumerate(current_fn_wav_list):
        fn_wav_list.append(fn_wav)
        class_label.append(current_class_label)
        file_num_in_class.append(k)

n_files: int = len(class_label)

# this vector includes a "counter" for each file within its class, we use it later ...
file_num_in_class = np.array(file_num_in_class)

unique_classes = sorted(list(set(class_label)))
print("All unique class labels (sorted alphabetically): ", unique_classes, '\n')
class_id = np.array([unique_classes.index(_) for _ in class_label])


#mel_specs erstellen ---------------------------------------------------------------------------------------------------


if mel_spec_diagram:
    mel_spec = msc.compute_mel_spec_for_audio_file(fn_wav_list[0])
    print("Shape of Mel-spectrogram (like diagram): ", mel_spec.shape, '\n')
    pl.figure()
    pl.imshow(mel_spec, origin="lower", aspect="auto", interpolation="None")
    pl.ylabel('Mel frequency bands')
    pl.xlabel('Time frames')
    pl.show()


all_mel_specs = []
for i, fn_wav in enumerate(fn_wav_list):
    all_mel_specs.append(msc.compute_mel_spec_for_audio_file(fn_wav_list[i]))

print("We have {} spectrograms of shape {}".format(len(all_mel_specs), all_mel_specs[0].shape))


# step 1: transpose all spectrograms (flip rows and columns) and store them in a list
all_mel_specs: list = [spec.T for spec in all_mel_specs]
# let's check the new size:
print(all_mel_specs[0].shape)


# step 2:
feature_matrix = np.vstack(all_mel_specs)
print("Feature matrix shape: {}".format(feature_matrix.shape), '\n')


if mel_spec_diagram_inverse:
    pl.figure(figsize=(7, 10))
    pl.imshow(feature_matrix, aspect="auto", interpolation="None")
    pl.ylabel('Time frame')
    pl.xlabel('Mel frequency band')
    pl.show()


#Labeling with vectors -------------------------------------------------------------------------------------------------


n_files: int = len(all_mel_specs)

all_file_id: list|ndarray = []
all_class_id: list|ndarray = []
all_file_num_in_class: list|ndarray = []


for cur_file_id in range(n_files):
    # how many time frames does this example have?
    cur_n_frames = all_mel_specs[cur_file_id].shape[0]

    # create a vector with file_ids (all the same for all frames of the current spectrogram)
    cur_file_id_vec = np.ones(cur_n_frames) * cur_file_id

    # we'll do the same with the class ID associated with the current file
    cur_class_id = np.ones(cur_n_frames) * class_id[cur_file_id]

    # and again for the index of the file within each class (0 for the first example per class, 1 for the second ...)
    cur_file_num_in_class = np.ones(cur_n_frames) * file_num_in_class[cur_file_id]

    all_file_id.append(cur_file_id_vec)
    all_class_id.append(cur_class_id)
    all_file_num_in_class.append(cur_file_num_in_class)


# finally, let's concatenate them to two large arrayss
all_file_id = np.concatenate(all_file_id)
all_class_id = np.concatenate(all_class_id)
all_file_num_in_class = np.concatenate(all_file_num_in_class)

print("Length of our file_id vector: ", len(all_file_id), '\n')


if file_id_diagram:
    pl.figure(figsize=(8,4))
    pl.subplot(2,1,1)
    pl.plot(all_file_id, label='File ID')
    pl.plot(all_file_num_in_class, label='File number per class')
    pl.title('File ID per feature vector')
    pl.legend()
    pl.xlabel('')
    pl.xlim(0, len(all_file_id)-1)
    pl.subplot(2,1,2)
    pl.title('Class ID per feature vector')
    pl.plot(all_class_id)
    pl.xlim(0, len(all_class_id)-1)
    pl.tight_layout()
    pl.show()


#Testdatensatz aufbereiten ---------------------------------------------------------------------------------------------


is_train = np.where(all_file_num_in_class >= 2)[0]
is_test = np.where(all_file_num_in_class <= 1)[0]

print("Our feature matrix is split into {} training examples and {} test examples.".format(len(is_train), len(is_test)), '\n')

X_train = feature_matrix[is_train, :]
y_train = all_class_id[is_train]
X_test = feature_matrix[is_test, :]
y_test = all_class_id[is_test]

print("Let's look at the dimensions:")
print('X_train: ', X_train.shape)
print('y_train: ', y_train.shape)
print('X_test:  ', X_test.shape)
print('y_test:  ', y_test.shape, '\n')

# zu Einheitsvektor normalisen
scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)


y_train_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_train.reshape(-1, 1))
y_test_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_test.reshape(-1, 1))

print("First four entries as class IDs:\n{}\n".format(y_train[:4]))
print("First four entries as one-hot-encoded vector:\n{}\n".format(y_train_transformed[:4, :]))

print("Shape of one-hot-encoded training targets: {}".format(y_train_transformed.shape))
print("Shape of one-hot-encoded test targets:     {}".format(y_test_transformed.shape), '\n')


# let's recall the dimension of our features:
n_input_dims = X_train_norm.shape[1]
print("Our input features have {} dimensions.".format(n_input_dims))

# ... and the number of classes:
n_classes = y_train_transformed.shape[1]
print("We want to classify between {} classes.".format(n_classes), '\n')


#Model trainieren ------------------------------------------------------------------------------------------------------


model = Sequential()
model.add(Input(shape=(n_input_dims,)))  # Replace input_dim with an Input layer (it's better for keras)
#model.add(Dense(128, input_dim=n_input_dims, activation='relu'))
model.add(Dense(128, activation='relu'))
for i in range(3): #number of hidden layers
    model.add(Dropout(.3)) #sets x % of the Neurons to zero to minimise errors
    model.add(Dense(64, activation='relu'))

# let's add the final layer (output layer)
model.add(Dense(n_classes, activation="softmax"))

# model.compile(loss='mean_squared_error', optimizer='softmax', metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Let's have a look on our model:
model.summary()
print('\n')
history = model.fit(X_train_norm, y_train_transformed, epochs=50, batch_size=32, verbose=2) #epochs = training rounds with batch_size samples each, verbose is what is shown per round


#accuracy --------------------------------------------------------------------------------------------------------------


if accuracy_diagram:
    pl.figure(figsize=(8, 4))
    pl.subplot(2,1,1)

    pl.plot(history.history['loss'])
    pl.ylabel('Loss')
    pl.subplot(2,1,2)
    pl.plot(history.history['accuracy'])
    pl.ylabel('Accuracy (Training Set)')
    pl.xlabel('Epoch')
    pl.show()

y_test_pred = model.predict(X_test_norm)

print('\n', "Shape of the predictions:     {}".format(y_test_pred.shape))

# The model outputs in each row 5 probability values (they always add to 1!) for each class.
# We want take the class with the highest probability as prediction!

y_test_pred = np.argmax(y_test_pred, axis=1)
print("Shape of the predictions now: {}".format(y_test_pred.shape))

# Accuracy
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy score:              ", accuracy, '\n')

ticks = np.arange(5)
if confusion_matrix_diagram:
    pl.figure(figsize=(3,3))
    cm = confusion_matrix(y_test, y_test_pred).astype(np.float32)
    # normalize to probabilities
    for i in range(cm.shape[0]):
        if np.sum(cm[i, :]) > 0:
            cm[i, :] /= np.sum(cm[i, :])  # by dividing through the sum, we convert counts to probabilities
    pl.imshow(cm)

    pl.xticks(ticks, unique_classes)
    pl.yticks(ticks, unique_classes)
    pl.show()


x, fs = librosa.load(fn_wav_list[-1])
ipd.display(ipd.Audio(data=x, rate=fs))

# extract the features
mel_spec = msc.compute_mel_spec_for_audio_file(fn_wav_list[-1])

# transpose it to a feature matrix
feat_mat = mel_spec.T

# normalize the features
feat_mat = scaler.transform(feat_mat)

# compute model predictions
class_probs = model.predict(feat_mat)

# let's visualize the spectrogram and the predictions
if specto_predictions_diagram:
    pl.figure(figsize=(12, 5))
    pl.subplot(2,1,1)
    pl.imshow(mel_spec, origin="lower", aspect="auto", interpolation="None")
    pl.subplot(2,1,2)
    pl.imshow(class_probs.T, origin="lower", aspect="auto", interpolation="None")
    pl.yticks(ticks, unique_classes)
    pl.tight_layout()
    pl.show()

