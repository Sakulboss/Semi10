import numpy as np
#from numba.cpython.randomimpl import seed_impl
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def training_data(data: tuple, setting: dict, logger) -> tuple:
    """
    This function prepares the training data: It splits the data into training and test data
    Args:
        data:     Data from the previous mel_specs function.
        setting:  Main settings like the type of dataset and injection of other labeled data
        logger:   The logger for logging.

    Returns:
        - input_shape: shape of the input data for the model
        - n_classes: number of classes
        - ... training and test datasets
        - data[4]: class names
    """

    # Initialize the working directory and variables
    #segment_file_mod_id  = setting.get('segment_file_mod_id', data[0]) --old unused
    segment_list         = setting.get('segment_list', data[1])
    segment_class_id     = setting.get('segment_class_id', data[2])
    test_size            = setting.get('test_size', 0.3)
    #seed                = setting.get('seed', )                        --unused due to misunderstanding of seed
    event_ratio          = setting.get('swarm_event_ratio', 0.5)        #Ratio of swarm_event 0.5 -> swarm_event equals no_swarm

    #np.random.seed(seed)

    # Indices of classes
    idx_swarm = np.where(segment_class_id == 1)[0]
    idx_no = np.where(segment_class_id == 0)[0]

    # Choose the minimum; important for the following case: 600 Swarm_data & 500 non_swarm_data; we choose 500 samples from each and distribute them; determined by: test_size
    min_class_count = min(len(idx_swarm), len(idx_no))

    # Choose the the indices for the minimum
    idx_swarm = idx_swarm[:min_class_count]
    idx_no = idx_no[:min_class_count]

    # Quantity of swarm & non swarm data; determined by: event_ratio
    test_count_swarm = int(min_class_count * test_size * event_ratio)
    test_count_no_event = int(min_class_count * test_size * (1 - event_ratio))

    # Choosing random test/train indices for: swarm_event
    test_swarm = np.random.choice(idx_swarm, size=test_count_swarm, replace=False)
    train_swarm = np.setdiff1d(idx_swarm, test_swarm)

    # Choosing random test/train indices for: no_event
    test_no = np.random.choice(idx_no, size=test_count_no, replace=False)
    train_no = np.setdiff1d(idx_no, test_no)

    # Create an empty boolean array with length of segment_class_id
    is_train = np.zeros(segment_class_id.shape[0], dtype=bool)
    is_test = np.zeros(segment_class_id.shape[0], dtype=bool)

    # Configure boolean arrays
    is_train[train_swarm] = True
    is_train[train_no] = True
    is_test[test_swarm] = True
    is_test[test_no] = True


    # Finally create training & test datasets
    x_train = segment_list[is_train, :, :]
    y_train = segment_class_id[is_train]

    x_test = segment_list[is_test, :, :]
    y_test = segment_class_id[is_test]

    #--------------------------------------------------------------
    logger.info(f'Ratio of test data: set value {test_size}; is value {len(x_test) / len(x_train)}')
    logger.info(f'Ratio of test data: set value {test_size}; is value {len(y_test) / len(y_train)}')
    logger.info(f'train_data is made of {np.sum(y_train == 1)} random swarm mels and {np.sum(y_train == 0)} random non swarm mels.')
    logger.info(f'test_data  is made of {np.sum(y_test == 1)} random swarm mels and {np.sum(y_test == 0)} random non swarm mels.')
    #---------------------------------------------------------------

    # Create empty norm-arrays for further processing
    x_train_norm = np.zeros_like(x_train)
    x_test_norm = np.zeros_like(x_test)

    #print("Train set:", x_train.shape, y_train.shape)      --only for checking
    #print("Test set:", x_test.shape, y_test.shape)
    #print("Norm arrays:", x_train_norm.shape, x_test_norm.shape)

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

    # The input shape is the "time-frequency shape" of our segments with the number of channels, which is 1 (needed for the model -> channel dimension (e.g., a rgb picture))
    input_shape = x_train_norm.shape[1:]
    n_classes = y_train_transformed.shape[1]

    try:
        assert n_classes == data[4] # Check if the number of classes is the same as in the data
    except AssertionError:
        logger.debug(f"Number of classes in data: {n_classes} vs. Number of classes in dir structure: {data[4]} - please correct")

    return input_shape, n_classes, x_train_norm, y_train_transformed, x_test_norm, y_test_transformed, y_test, data[3], data[4]
