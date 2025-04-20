import numpy as np
#import sklearn as skl
import os
#import matplotlib
import librosa
import matplotlib.pyplot as pl
#import platform
import IPython.display as ipd
import wget
import zipfile
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


X = np.array(((1,1),
              (1, 1.5),
              (1.5, 1),
              (1, 3),
              (1, 4),
              (1.5, 4),
              (3, 1),
              (4, 1),
              (4, 1.5)))

# let's look at the shape of X
n_items, n_dims = X.shape
print("We have {} data points and {} feature dimensions".format(n_items, n_dims), '\n')

print(X, '\n')

pl.figure(figsize=(4, 4))
pl.plot(X[:, 0], X[:, 1], 'bo')  # first argument: X[:, 0] represents a vector with all
                                       # values of the first feature dimension
                                       # second argument: X[:, 1] ... second feature dimension
                                       # 'bo' includes 'b' for blue and 'o' for circle-shaped markers
pl.xlabel('Feature dimension 1')
pl.ylabel('Feature dimension 2')
pl.show()

from sklearn.cluster import KMeans
kmeans_clustering = KMeans(n_clusters=3)

kmeans_clustering.fit(X)

cluster_numbers = kmeans_clustering.predict(X)
print(cluster_numbers, '\n')

pl.figure(figsize=(4, 4))
marker_colors = ['r', 'b', 'g', 'y']  # red, blue, and green markers
marker_shape = ['o', 's', 'd', '+']  # circle, square, diamond shaped markers

# now we iterate over all data points and plot them one-by-one
for i in range(n_items):
    # which cluster number is this data point assigned to?
    current_cluster_num = cluster_numbers[i]
    pl.plot(X[i, 0], X[i, 1],
            marker=marker_shape[current_cluster_num],
            color=marker_colors[current_cluster_num])
pl.xlabel('Feature dimension 1')
pl.ylabel('Feature dimension 2')
pl.show()

X_train = np.array(((1,1),
                    (1, 1.5),
                    (1.5, 1),
                    (1, 3),
                    (1, 4),
                    (1.5, 4),
                    (3, 1),
                    (4, 1),
                    (4, 1.5)))

y_train = np.array((0, 0, 0, 1, 1, 1, 2, 2, 2))

# number of training examples
n_train = len(y_train)

print(X_train, '\n')

X_test = np.array(((1.2, 1.2),
                   (1.7, 2.7),
                   (2, 2),
                   (2, 3),
                   (3, 3),
                   (3.5, 1.5)))

y_test = np.array((0, 1, 0, 1, 2, 2))

# number of test examples
n_test = len(y_test)

pl.figure(figsize=(6, 4))

# again, unique marker colors and shapes for our classes
marker_colors = ['r', 'b', 'g']
marker_shape = ['o', 's', 'd']

# plot the training dataset
for i in range(n_train):
    # which cluster number is this data point assigned to?
    current_cluster_num = cluster_numbers[i]
    pl.plot(X_train[i, 0], X_train[i, 1],
            marker=marker_shape[current_cluster_num],
            color=marker_colors[current_cluster_num])

# plot the test dataset (but without labels for now)
for i in range(n_test):
    pl.plot(X_test[i, 0], X_test[i, 1], 'ko', markersize=10)
    pl.text(X_test[i, 0] + .05, X_test[i, 1] + .2, '?')

pl.title('Training set (red, green, blue) + Test set (black)', fontsize=8)
pl.xlabel('Feature dimension 1')
pl.ylabel('Feature dimension 2')
pl.show()

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_test_predict = classifier.predict(X_test)

print("Here are the class predictions for the test set: ", y_test_predict)
print("Here are the true class indices for the test set: ", y_test, '\n')

accuracy = accuracy_score(y_test, y_test_predict)

print("Accuracy score = ", accuracy, '\n')

conf_mat = confusion_matrix(y_test, y_test_predict)

print("Confusion matrix: ")
print(conf_mat, '\n')

fig = pl.figure(figsize=(4,4))
ConfusionMatrixDisplay.from_predictions(y_test, y_test_predict, ax=fig.gca(), colorbar=False)
pl.show()

if not os.path.isfile('../Sound_processing/animal_sounds.zip'):
    print('Please wait a couple of seconds ...')
    wget.download(
        #'https://github.com/karoldvl/ESC-50/archive/master.zip',
        'https://github.com/machinelistening/machinelistening.github.io/blob/master/animal_sounds.zip?raw=true',
        out='_animal_sounds.zip', bar=None)
    print('_animal_sounds.zip downloaded successfully ...')
else:
    print('Files already exist!')

if not os.path.isdir('../Sound_processing/_animal_sounds'):
    print("Let's unzip the file ... ")
    assert os.path.isfile('../Sound_processing/animal_sounds.zip')
    with zipfile.ZipFile('../Sound_processing/animal_sounds.zip', 'r') as f:
        # Entpacke alle Inhalte in das angegebene Verzeichnis
        f.extractall('.')
    assert os.path.isdir('../Sound_processing/_animal_sounds')
    print("All done :)", '\n')

dir_dataset = '../Sound_processing/_animal_sounds'
sub_directories = glob.glob(os.path.join(dir_dataset, '*'))

print(sub_directories)

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
print('Here is our list of audio files, sorted by sound classes:')
for i in range(n_files):
    print(class_label[i], '-', fn_wav_list[i])

# this vector includes a "counter" for each file within its class, we use it later ...
file_num_in_class = np.array(file_num_in_class)
print(file_num_in_class)

for i in range(5):
    idx = 5*i  # always take the first one per class
    x, fs = librosa.load(fn_wav_list[idx])
    print(class_label[idx])
    ipd.display(ipd.Audio(data=x, rate=fs))

print(class_label)

unique_classes = sorted(list(set(class_label)))
print("All unique class labels (sorted alphabetically)", unique_classes)

# now we can iterate over all files and look for the index of its class in this list

print(unique_classes.index('cat'))
print(unique_classes.index('cow'))

class_id = np.array([unique_classes.index(_) for _ in class_label])

print("Class IDs of all files", class_id)

x, fs = librosa.load(fn_wav_list[0])
mfcc = librosa.feature.mfcc(y=x, n_mfcc=13)
print("Shape of MFCC matrix", mfcc.shape)
# let's average it over time to get a global feature vector which measures
# the overall timbre of the audio file
feature_vector = np.mean(mfcc, axis=1)
print("Shape of our final feature vector", feature_vector.shape)

pl.figure()
pl.imshow(mfcc, origin="lower", aspect="auto", interpolation="None")
pl.show()

feature_matrix = []
for i, fn_wav in enumerate(fn_wav_list):
    x, fs = librosa.load(fn_wav)
    mfcc = librosa.feature.mfcc(y=x, n_mfcc=13)
    # store current feature vector
    feature_matrix.append(np.mean(mfcc, axis=1))
# by now, feature_matrix is just a list of one-dimensional numpy arrays.
# let's convert it into a two-dimensional array (feature matrix)
feature_matrix = np.vstack(feature_matrix) # vertically stacking - row-by-row
print("Final shape of our feature matrix", feature_matrix.shape)

print("Remember how it looks like:", file_num_in_class)  # starts at 0 for the first file in each class, etc...

is_train = np.where(file_num_in_class <= 2)[0]
is_test = np.where(file_num_in_class >= 3)[0]

print("Indices of the training set items:", is_train)
print("Indices of the test set items:", is_test)

X_train = feature_matrix[is_train, :]
y_train = class_id[is_train]
X_test = feature_matrix[is_test, :]
y_test = class_id[is_test]

print("Let's look at the dimensions")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

scaler = StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)

print("Let's check the mean and standard deviation values over our training set feature matrix for each feature dimension BEFORE...")
print(np.mean(X_train, axis=0))
print(np.std(X_train, axis=0))
print("and AFTER the normalization ...")
print(np.mean(X_train_norm, axis=0))
print(np.std(X_train_norm, axis=0))


classifier = RandomForestClassifier(n_estimators=30)
classifier.fit(X_train_norm, y_train)

# apply the normalization learnt from the training set
X_test_norm = scaler.transform(X_test)
y_test_predict = classifier.predict(X_test_norm)

print("Here are the class predictions for the test set: ", y_test_predict)
print("Here are the true class indices for the test set: ", y_test)

accuracy = accuracy_score(y_test, y_test_predict)
print("Accuracy score = ", accuracy)



fig = pl.figure(figsize=(5,4))
print(y_test, y_test_predict)
ConfusionMatrixDisplay.from_predictions(y_test, y_test_predict, ax=fig.gca(), colorbar=False, display_labels=unique_classes)
ticks = np.arange(5)
pl.xticks(ticks, unique_classes)
pl.yticks(ticks, unique_classes)
pl.show()