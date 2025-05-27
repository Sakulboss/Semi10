import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def refine_data(data: tuple) -> tuple:
    """
    Args:
        data: tuple with mel spectrograms and class ids from the previous step
    Returns:
        y_one_hot:        list with OneHotEncoded Class-Ids
        segment_class_id: list with Class Ids as 1d-array
        x_all:            list with normalized samples and additional dimension

    """
    segment_list = data[1]
    segment_class_id = data[2]

    # Create an empty array, like segment_list
    x_all = np.zeros_like(segment_list)

    # Normalize samples
    for i in range(segment_list.shape[0]):
        x_all[i, :, :] = StandardScaler().fit_transform(segment_list[i, :, :])

    # Add channel dimension for CNN
    if len(x_all.shape) == 3:
        x_all = np.expand_dims(x_all, 1)

    # OneHotEncode Class-Ids
    y_one_hot = OneHotEncoder(sparse_output=False).fit_transform(segment_class_id.reshape(-1, 1))

    return y_one_hot, segment_class_id, x_all
