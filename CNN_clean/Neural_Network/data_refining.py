import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def training_data(data: tuple, setting: dict, logger) -> tuple:
    segment_list = setting.get('segment_list', data[1])
    segment_class_id = setting.get('segment_class_id', data[2])
    test_size = setting.get('test_size', 0.3)
    event_ratio = setting.get('swarm_event_ratio', 0.5)

    # Normierung aller Samples
    x_all = np.zeros_like(segment_list)
    for i in range(segment_list.shape[0]):
        x_all[i, :, :] = StandardScaler().fit_transform(segment_list[i, :, :])

    # Dimension hinzufügen für CNN (Channels)
    if len(x_all.shape) == 3:
        x_all = np.expand_dims(x_all, 1)

    y_all = segment_class_id
    y_all_oh = OneHotEncoder(sparse_output=False).fit_transform(y_all.reshape(-1, 1))
    return y_all_oh, y_all, x_all
#--------------------------------------------------------------------------------------------------------------------------------------------
    '''
    # Jetzt Split logik nach Verhältnissen (aber alles ist schon normiert + encoded)
    idx_swarm = np.where(y_all == 1)[0]
    idx_no_swarm = np.where(y_all == 0)[0]

    min_class_count = min(len(idx_swarm), len(idx_no_swarm))
    idx_swarm = idx_swarm[:min_class_count]
    idx_no_swarm = idx_no_swarm[:min_class_count]

    test_count_swarm = int(min_class_count * test_size * event_ratio)
    test_count_no = int(min_class_count * test_size * (1 - event_ratio))

    # Test/Train Indexwahl
    test_swarm = np.random.choice(idx_swarm, size=test_count_swarm, replace=False)
    train_swarm = np.setdiff1d(idx_swarm, test_swarm)

    test_no = np.random.choice(idx_no_swarm, size=test_count_no, replace=False)
    train_no = np.setdiff1d(idx_no_swarm, test_no)

    # Finales Shuffle & Auswahl
    train_idx = np.concatenate([train_swarm, train_no])
    test_idx = np.concatenate([test_swarm, test_no])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    x_train_norm = x_all[train_idx]
    y_train_transformed = y_all_oh[train_idx]

    x_test_norm = x_all[test_idx]
    y_test_transformed = y_all_oh[test_idx]
    y_test = y_all[test_idx]  # original labels für spätere Auswertung

    input_shape = x_train_norm.shape[1:]
    n_classes = y_train_transformed.shape[1]

    # ---------------------------------------------------------------
    logger.debug(f"Train set:   {x_train_norm.shape}, {y_train_transformed.shape}")
    logger.debug(f"Test set:    {x_test_norm.shape}, {y_test_transformed.shape}")
    logger.debug(f"Norm arrays: {x_train_norm.shape}, {x_test_norm.shape}")
    # ---------------------------------------------------------------

    # --------------------------------------------------------------
    logger.info(f"Ratio of test data: set value {test_size}; is value {len(x_test_norm) / len(x_train_norm)}")
    logger.info(f"Ratio of training data: set value {test_size}; is value {len(y_test_transformed) / len(y_train_transformed)}")
    logger.info(f"train_data is made of {np.sum(np.argmax(y_train_transformed, axis=1) == 1)} random swarm mels and {np.sum(np.argmax(y_train_transformed, axis=1) == 0)} random non swarm mels.")
    logger.info(f"test_data  is made of {np.sum(np.argmax(y_test_transformed, axis=1) == 1)} random swarm mels and {np.sum(np.argmax(y_test_transformed, axis=1) == 0)} random non swarm mels.")
    # ---------------------------------------------------------------

    try:
        assert n_classes == data[4]
    except AssertionError:
        logger.debug(f"Mismatch in class count: {n_classes} vs {data[4]}")

    return input_shape, n_classes, x_train_norm, y_train_transformed, x_test_norm, y_test_transformed, y_test, data[3], data[4]










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
    segment_list         = setting.get('segment_list', data[1])
    segment_class_id     = setting.get('segment_class_id', data[2])
    test_size            = setting.get('test_size', 0.3)
    event_ratio          = setting.get('swarm_event_ratio', 0.5)        #Ratio of swarm_event 0.5 -> swarm_event equals no_swarm

    #np.random.seed(seed)

    # Indices of classes
    idx_swarm = np.where(segment_class_id == 1)[0]
    idx_no = np.where(segment_class_id == 0)[0]

    # Choose the minimum; important for the following case: 600 Swarm_data & 500 non_swarm_data; we choose 500 samples from each and distribute them; determined by: test_size
    min_class_count = min(len(idx_swarm), len(idx_no))

    # Choose the indices for the minimum
    idx_swarm = idx_swarm[:min_class_count]
    idx_no_event = idx_no[:min_class_count]

    # Quantity of swarm & non swarm data; determined by: event_ratio
    test_count_swarm = int(min_class_count * test_size * event_ratio)
    test_count_no_event = int(min_class_count * test_size * (1 - event_ratio))

    # Choosing random test/train indices for: swarm_event
    test_swarm = np.random.choice(idx_swarm, size=test_count_swarm, replace=False)
    train_swarm = np.setdiff1d(idx_swarm, test_swarm)

    # Choosing random test/train indices for: no_event
    test_no_event = np.random.choice(idx_no, size=test_count_no_event, replace=False)
    train_no_event = np.setdiff1d(idx_no_event, test_no_event)

    # Create an empty boolean array with length of segment_class_id
    is_train = np.zeros(segment_class_id.shape[0], dtype=bool)
    is_test = np.zeros(segment_class_id.shape[0], dtype=bool)

    # Configure boolean arrays
    is_train[train_swarm] = True
    is_train[train_no_event] = True
    is_test[test_swarm] = True
    is_test[test_no_event] = True


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

    #---------------------------------------------------------------
    logger.debug(f"Train set:   {x_train.shape}, {y_train.shape}")
    logger.debug(f"Test set:    {x_test.shape}, {y_test.shape}")
    logger.debug(f"Norm arrays: {x_train_norm.shape}, {x_test_norm.shape}")
    #---------------------------------------------------------------

    # Transform the data
    for i in range(x_train.shape[0]):
        x_train_norm[i, :, :] = StandardScaler().fit_transform(x_train[i, :, :])

    for i in range(x_test.shape[0]):
        x_test_norm[i, :, :] = StandardScaler().fit_transform(x_test[i, :, :])

    # Encode the labels with OneHotEncoder so that they represent the wanted output by the CNN {(2,5) -> ((0,0,1,0,0,0), (0,0,0,0,0,1))}
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
'''