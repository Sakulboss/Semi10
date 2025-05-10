import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def training_data(data: tuple, setting: dict, ) -> tuple:
    """
    This function prepares the training data: It splits the data into training and test data
    Args:
        data:     Data from the previous mel_specs function.
        setting:  Main settings like the type of dataset and injection of other labeled data.

    Returns:


    """

    model: str           = setting.get('model', 'torch')
    segment_file_mod_id  = setting.get('segment_file_mod_id', data[0])
    segment_list         = setting.get('segment_list', data[1])
    segment_class_id     = setting.get('segment_class_id', data[2])
    printing             = setting.get('printing', False)

    if model not in ['torch', 'tf']: raise ValueError('Got unknown model name.')

    is_train = np.where(segment_file_mod_id <= 2)[0]
    is_test = np.where(segment_file_mod_id >= 3)[0]

    if printing: print("Our feature matrix is split into {} training examples and {} test examples".format(len(is_train), len(is_test)))

    x_train = segment_list[is_train, :, :]
    y_train = segment_class_id[is_train]
    x_test = segment_list[is_test, :, :]
    y_test = segment_class_id[is_test]

    x_train_norm = np.zeros_like(x_train)
    x_test_norm = np.zeros_like(x_test)

    for i in range(x_train.shape[0]):
        x_train_norm[i, :, :] = StandardScaler().fit_transform(x_train[i, :, :])

    for i in range(x_test.shape[0]):
        x_test_norm[i, :, :] = StandardScaler().fit_transform(x_test[i, :, :])

    y_train_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_train.reshape(-1, 1))
    y_test_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_test.reshape(-1, 1))

    if len(x_train_norm.shape) == 3 and model == 'torch':
        x_train_norm = np.expand_dims(x_train_norm, 1)
        x_test_norm = np.expand_dims(x_test_norm, 1)
    elif len(x_train_norm.shape) == 3 and model == 'tf':
        x_train_norm = np.expand_dims(x_train_norm, -1)
        x_test_norm = np.expand_dims(x_test_norm, -1)

    if printing: print(f"Let's check if we have four dimensions. New shapes: {x_train_norm.shape} & {x_test_norm.shape}")

    # The input shape is the "time-frequency shape" of our segments + the number of channels
    input_shape = x_train_norm.shape[1:]

    # Get the number of classes:
    n_classes = y_train_transformed.shape[1]

    return input_shape, n_classes, x_train_norm, y_train_transformed, x_test_norm, y_test_transformed, y_test, data[3], data[4]