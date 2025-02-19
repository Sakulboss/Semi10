import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def training_data(data, setting):
    segment_file_mod_id = setting.get('segment_file_mod_id', data[0])
    segment_list = setting.get('segment_list', data[1])
    segment_class_id = setting.get('segment_class_id', data[2])
    printing = setting.get('printing', False)

    is_train = np.where(segment_file_mod_id <= 2)[0]
    is_test = np.where(segment_file_mod_id >= 3)[0]

    if printing: print("Our feature matrix is split into {} training examples and {} test examples".format(len(is_train), len(is_test)))

    X_train = segment_list[is_train, :, :]
    y_train = segment_class_id[is_train]
    X_test = segment_list[is_test, :, :]
    y_test = segment_class_id[is_test]

    if printing: print("Let's look at the dimensions")
    if printing: print(X_train.shape)
    if printing: print(y_train.shape)
    if printing: print(X_test.shape)
    if printing: print(y_test.shape)

    X_train_norm = np.zeros_like(X_train)
    X_test_norm = np.zeros_like(X_test)

    for i in range(X_train.shape[0]):
        X_train_norm[i, :, :] = StandardScaler().fit_transform(X_train[i, :, :])

    for i in range(X_test.shape[0]):
        X_test_norm[i, :, :] = StandardScaler().fit_transform(X_test[i, :, :])

    y_train_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_train.reshape(-1, 1))
    y_test_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_test.reshape(-1, 1))

    #if len(X_train_norm.shape) == 3:
    X_train_norm = np.expand_dims(X_train_norm, 1)
    X_test_norm = np.expand_dims(X_test_norm, 1)

    #else:
    #    if printing: print("We already have four dimensions")

    if printing: print(f"Let's check if we have four dimensions. New shapes: {X_train_norm.shape} & {X_test_norm.shape}")

    # The input shape is the "time-frequency shape" of our segments + the number of channels
    # Make sure to NOT include the first (batch) dimension!
    input_shape = X_train_norm.shape[1:]

    # Get the number of classes:
    n_classes = y_train_transformed.shape[1]

    return input_shape, n_classes, X_train_norm, y_train_transformed, X_test_norm, y_test_transformed, y_test, data[3], data[4]