from collections import Counter

import numpy as np
import pandas as pd


def value_counts(y):
    """value_counts return additional Counter information

    Parameters
    ----------
    y : list, pandas series
        list of labels
    """
    counter = Counter(y)
    for k, v in counter.items():
        per = v / len(y) * 100
        print("Class=%s, Count=%d, Percentage=%.3f%%" % (str(k), v, per))


def flatten_dataframe(data: pd.DataFrame, freq=10) -> np.ndarray:
    """flatten_dataframe
    Flatten pandas DataFrame

    Parameters
    ----------
    data : pd.DataFrame
        input dataframe
    freq : int, optional
        frequency of data (incremental), by default 10

    Returns
    -------
    np.ndarray
        array of flattened dataframe values
    """
    select = [i for i in range(data.shape[0]) if i % freq == 0]
    new_data = data.loc[select].reset_index(drop=True)
    return new_data.transpose().to_numpy().flatten()
