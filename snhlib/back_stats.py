from dataclasses import dataclass

import numpy as np
from scipy.stats import ks_2samp, norm
import matplotlib.pyplot as plt
import pandas as pd


class Significant:
    def __init__(self, verbose=0, p=0.05):
        self.verbose = verbose
        self.pvalue = p

    def KolmogorovSmirnov(self, kontrol, test):
        val = ks_2samp(kontrol, test)
        res = "rejected" if val < self.pvalue else "accepted"
        if self.verbose != 0:
            print(f"H0 was {res} with K-S value {val}")
        return val, res

    def PSI(self, kontrol, test):
        val = self.calculate_psi(kontrol, test)

        if val < 0.1:
            res = "no significant population change"
        elif val < 0.2:
            res = "moderate population change"
        elif val >= 0.2:
            res = "significant population change"

        if self.verbose != 0:
            print(f"PSI value {val} with {res}")
        return val, res

    @staticmethod
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

        def psi(expected_array, actual_array, buckets):
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
                breakpoints = scale_range(
                    breakpoints, np.min(expected_array), np.max(expected_array)
                )
            elif buckettype == "quantiles":
                breakpoints = np.stack(
                    [np.percentile(expected_array, b) for b in breakpoints]
                )

            expected_percents = np.histogram(expected_array, breakpoints)[0] / len(
                expected_array
            )
            actual_percents = np.histogram(actual_array, breakpoints)[0] / len(
                actual_array
            )

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

            psi_value = np.sum(
                sub_psi(expected_percents[i], actual_percents[i])
                for i in range(0, len(expected_percents))
            )

            return psi_value

        if len(expected.shape) == 1:
            psi_values = np.empty(len(expected.shape))
        else:
            psi_values = np.empty(expected.shape[axis])

        for i in range(0, len(psi_values)):
            if len(psi_values) == 1:
                psi_values = psi(expected, actual, buckets)
            elif axis == 0:
                psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
            elif axis == 1:
                psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

        return psi_values

    @staticmethod
    def interpret_cohens_d(val):
        val = abs(val)
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

    def cohend(self, kontrol, test):
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
        res = (u1 - u2) / s
        size = self.interpret_cohens_d(res)
        return res, size


@dataclass
class AUC_confidence_interval:
    n1: int
    n2: int
    auc: float
    alpha: float = 0.05

    @property
    def zcrit_(self):
        return norm.ppf(1 - self.alpha/2)

    @property
    def se_(self):
        q0 = self.auc * (1 - self.auc)
        q1 = self.auc / (2 - self.auc) - self.auc**2
        q2 = 2 * self.auc**2 / (1 + self.auc) - self.auc**2
        return np.sqrt(
            (q0 + (self.n1 - 1) * q1 + (self.n2 - 1) * q2) / (self.n1 * self.n2)
        )

    @property
    def interval_(self):
        lower = self.auc - self.zcrit_ * self.se_
        upper = self.auc + self.zcrit_ * self.se_
        return lower, upper


class Bootstrap:
    def __init__(self, random_state=None, estimator=np.mean, bootsample=1000) -> None:
        self.random_state=random_state
        self.estimator=estimator
        self.bootsample=bootsample

    def generate_bootstrap(self, arr):
        np.random.seed(self.random_state)
        return [self.estimator(np.random.choice(arr, len(arr))) for _ in range(self.bootsample)]

    def plot_bootstrap(self, arr):
        boot = self.generate_bootstrap(arr)

        plt.style.use("bmh")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.boxplot(
            boot,
            widths=0.5,
            whis=[2.5, 97.5],
            showfliers=False,
            patch_artist=True,
            boxprops=dict(linewidth=3.0, color="grey"),
            whiskerprops=dict(linewidth=3.0, color="grey"),
            vert=False,
            capprops=dict(linewidth=2.0, color="grey"),
            medianprops=dict(linewidth=2.0, color="yellow"),
        )

        ax.set_aspect(1)
        ax.set_xlabel(r"$\bar{X}$", fontsize=20)
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.set_title("Bootstrap 95% confidence interval of mean", fontsize=20)
        ax.scatter(
            arr,
            [1, 1, 1, 1, 1],
            facecolors="grey",
            edgecolors="k",
            zorder=10,
            label="Original Samples",
            s=100,
        )
        ax.legend(
            fontsize=15, fancybox=True, framealpha=1, shadow=True, borderpad=0.5, frameon=True
        )

        return fig

    def plot_hist(self, arr):
        boot = self.generate_bootstrap(arr)
        data = pd.Series(boot)

        data.plot.hist(grid=True, bins=20, rwidth=0.9, color="#607c8e")
        plt.xlabel("Sample")
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)
        return plt
