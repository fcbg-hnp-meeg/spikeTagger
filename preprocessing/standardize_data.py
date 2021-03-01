# standardize_data.py
import numpy as np

def standardize_data(data):  # shape==(pex,features)
    """
    Data standardization (data-mean)/sqrt(variance)
    Args:
        data:

    Returns:

    """
    print('--- --------------standardize_data-----------------', data.shape)
    # print(data)
    m = np.apply_along_axis(np.mean, 0, data)
    # print(m)
    sd = np.apply_along_axis(np.std, 0, data)  # luckily won t have any item==0
    # print('#zeros',sum(sd==0))
    # print('#Nan',sum(sd==np.nan))

    # print(sd)

    # centering
    centered = data - m[None, :]
    # print(centered)

    # scaling
    standardized = centered / sd[None, :]

    return (standardized)