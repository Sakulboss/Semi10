import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def training_data(data: tuple, setting: dict, logger) -> tuple:
    """
    This function prepares the training data: It splits the data into training and test data
    Args:
        data:     Data from the previous mel_specs function.
        setting:  Main settings like the type of dataset and injection of other labeled data.

    Returns:
        - input_shape: shape of the input data for the model
        - n_classes: number of classes
        - ... training and test datasets
        - data[4]: class names
    """

    # Initialize the working directory and variables
    segment_file_mod_id  = setting.get('segment_file_mod_id', data[0])
    segment_list         = setting.get('segment_list', data[1])
    segment_class_id     = setting.get('segment_class_id', data[2])
    printing             = setting.get('printing', False)
    test_size            = setting.get('test_size', 0.3)

    # Split the data into training and test data
    is_train = np.where(segment_file_mod_id <  test_size * 10)[0]
    is_test  = np.where(segment_file_mod_id >= test_size * 10)[0]

    logger.info(f"Our feature matrix is split into {len(is_train)} training examples and {len(is_test)} test examples")

    # Now the data itself is split with the indices
    x_train = segment_list[is_train, :, :]
    y_train = segment_class_id[is_train]
    x_test = segment_list[is_test, :, :]
    y_test = segment_class_id[is_test]

    x_train_norm = np.zeros_like(x_train)
    x_test_norm = np.zeros_like(x_test)

    # Transform the data
    for i in range(x_train.shape[0]):
        x_train_norm[i, :, :] = StandardScaler().fit_transform(x_train[i, :, :])

    for i in range(x_test.shape[0]):
        x_test_norm[i, :, :] = StandardScaler().fit_transform(x_test[i, :, :])

    # Encode the labels with OneHotEncoder so that they represent the wanted output by the CNN (2,5) -> ((0,0,1,0,0,0), (0,0,0,0,0,1))
    y_train_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_train.reshape(-1, 1))
    y_test_transformed = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(y_test.reshape(-1, 1))

    # Create the right dimensions
    if len(x_train_norm.shape) == 3:
        x_train_norm = np.expand_dims(x_train_norm, 1)
        x_test_norm = np.expand_dims(x_test_norm, 1)

    logger.info(f"Shapes of the train and test data: {x_train_norm.shape} & {x_test_norm.shape}")

    # The input shape is the "time-frequency shape" of our segments plus the number of channels, which is 1 (needed for the model -> channel dimension)
    input_shape = x_train_norm.shape[1:]
    n_classes = y_train_transformed.shape[1]

    assert n_classes == data[4] # Check if the number of classes is the same as in the data

    return input_shape, n_classes, x_train_norm, y_train_transformed, x_test_norm, y_test_transformed, y_test, data[3], data[4]