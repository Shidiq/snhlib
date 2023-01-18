from typing import Tuple

import numpy as np
from scipy.stats import ks_2samp


def kolmogorov_smirnov(kontrol: list, test: list, pvalue=0.05, verbose=0) -> tuple:
    """KolmogorovSmirnov
    KS hypothesis test

    Parameters
    ----------
    kontrol : list
    test : list
    pvalue : float, optional
        p-value, by default 0.05
    verbose : int, optional
        print output, by default 0

    Returns
    -------
    tuple
        KS-value, results
    """
    val = float(ks_2samp(kontrol, test))
    res = "rejected" if val < pvalue else "accepted"
    if verbose != 0:
        print(f"H0 was {res} with K-S value {val}")
    return val, res


def interpret_cohens_d(val: float) -> str:
    """interpret_cohens_d
    Interpreation of Cohen distance

    Parameters
    ----------
    val : float
        distance

    Returns
    -------
    str
        interpretation
    """
    val = abs(val)
    effect_size_interpretation = str()

    if 0 <= val < 0.1:
        effect_size_interpretation = "Very Small"
    elif 0.1 <= val < 0.35:
        effect_size_interpretation = "Small"
    elif 0.35 <= val < 0.65:
        effect_size_interpretation = "Medium"
    elif 0.65 <= val < 0.9:
        effect_size_interpretation = "Large"
    elif val >= 0.9:
        effect_size_interpretation = "Very Large"

    return effect_size_interpretation


def cohen_d(kontrol: list, test: list) -> Tuple:
    """cohend
    Calculate Cohen distance

    Parameters
    ----------
    kontrol : list
    test : list

    Returns
    -------
    Tuple
        distance, interpretation
    """
    d1 = kontrol
    d2 = test
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    distance = (u1 - u2) / s
    interpretation = interpret_cohens_d(distance)
    return distance, interpretation


def sub_psi(e_perc, a_perc):
    """Calculate the actual PSI value from comparing the values.
    Update the actual value to a very small number if equal to zero
    """
    if a_perc == 0:
        a_perc = 0.0001
    if e_perc == 0:
        e_perc = 0.0001

    value = (e_perc - a_perc) * np.log(e_perc / a_perc)
    return value


def psi(expected_array, actual_array, buckets, buckettype):
    """Calculate the PSI for a single variable

    Args:
    expected_array: numpy array of original values
    actual_array: numpy array of new values, same size as expected
    buckets: number of percentile ranges to bucket the values into

    Returns:
    psi_value: calculated PSI value
    """

    def scale_range(input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if buckettype == "bins":
        breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
    elif buckettype == "quantiles":
        breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

    expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
    actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

    psi_value = np.sum(
        sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))
    )

    return psi_value


def calculate_psi(expected, actual, buckettype="bins", buckets=10, axis=0):
    """Calculate the PSI (population stability index) across all variables

    Args:
    expected: numpy matrix of original values
    actual: numpy matrix of new values, same size as expected
    buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
    buckets: number of quantiles to use in bucketing variables
    axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
    psi_values: ndarray of psi values for each variable

    Author:
    Matthew Burke
    github.com/mwburke
    worksofchart.com
    """

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets, buckettype)
        elif axis == 0:
            psi_values[i] = psi(expected[:, i], actual[:, i], buckets, buckettype)
        elif axis == 1:
            psi_values[i] = psi(expected[i, :], actual[i, :], buckets, buckettype)

    return psi_values
