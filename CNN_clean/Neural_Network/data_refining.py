import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def refine_data(data: tuple, setting: dict) -> tuple:
    """

    Args:
        data: get the original data
        setting: get the settings from

    Returns:
        tuple consisting of
        y_all_oh: OneHotEncoded Array with Class-Ids
        y_all: Class Ids as 1d-array
        x_all: normalized samples with additional dimension

    """
    segment_list = setting.get('segment_list', data[1])
    segment_class_id = setting.get('segment_class_id', data[2])

    # create empty array, like segment_list
    x_all = np.zeros_like(segment_list)
    # normalize samples
    for i in range(segment_list.shape[0]):
        x_all[i, :, :] = StandardScaler().fit_transform(segment_list[i, :, :])

    # add dimension for CNN (Channels)
    if len(x_all.shape) == 3:
        x_all = np.expand_dims(x_all, 1)

    # class-Ids as 1d-array
    y_all = segment_class_id
    # OneHotEncode Class-Ids
    y_all_oh = OneHotEncoder(sparse_output=False).fit_transform(y_all.reshape(-1, 1))

    return y_all_oh, y_all, x_all
