import pandas as pd


def is_outlier(data, threshold=3):
    """
    Detects outliers in a pandas Series or DataFrame using z-normalization.

    Args:
        data: A pandas Series or DataFrame containing numerical data.
        threshold: The number of standard deviations above or below the mean to define an outlier (default: 3).

    Returns:
        A pandas Series of booleans with True indicating an outlier and False otherwise.
    """

    if isinstance(data, pd.Series):
        z_score = (data - data.mean()) / data.std()
    else:
        z_score = (data - data.mean(axis=0)) / data.std(axis=0)

    return abs(z_score) > threshold


def any_outlier(x):
    _any_outlier = any(x)
    _n_outlier = sum(x)
    _percentage = round(_n_outlier / len(x) * 100, 2)
    return _any_outlier, _n_outlier, _percentage


def summary_outlier(data):
    results = pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        for i, col in enumerate(list(data)):
            _out = is_outlier(data[col])
            _any_outlier, _n_outlier, _percentage = any_outlier(_out)

            results.loc[i, "feature"] = str(col)
            results.loc[i, "any outlier"] = bool(_any_outlier)
            results.loc[i, "number outlier"] = int(_n_outlier)
            results.loc[i, "percentage outlier"] = float(_percentage)

        return results
    else:
        raise ValueError("Input must Pandas Dataframe!")
