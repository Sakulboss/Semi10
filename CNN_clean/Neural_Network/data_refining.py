import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def training_data(data: tuple, setting: dict, logger) -> tuple:
    segment_list = setting.get('segment_list', data[1])
    segment_class_id = setting.get('segment_class_id', data[2])

    # Create empty array, like segment_list
    x_all = np.zeros_like(segment_list)
    # Normalize samples
    for i in range(segment_list.shape[0]):
        x_all[i, :, :] = StandardScaler().fit_transform(segment_list[i, :, :])

    # Add dimension for CNN (Channels)
    if len(x_all.shape) == 3:
        x_all = np.expand_dims(x_all, 1)

    # Class-Ids as 1d-array
    y_all = segment_class_id
    # OneHotEncode Class-Ids
    y_all_oh = OneHotEncoder(sparse_output=False).fit_transform(y_all.reshape(-1, 1))

    return y_all_oh, y_all, x_all
#--------------------------------------------------------------------------------------------------------------------------------------------
