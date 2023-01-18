import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Bootstrap:
    def __init__(self, random_state=None, estimator=np.mean, bootsample=1000) -> None:
        self.random_state = random_state
        self.estimator = estimator
        self.bootsample = bootsample

    def generate_bootstrap(self, arr):
        np.random.seed(self.random_state)
        return [self.estimator(np.random.choice(arr, len(arr))) for _ in range(self.bootsample)]

    def plot_bootstrap(self, arr):
        boot = self.generate_bootstrap(arr)

        plt.style.use("bmh")
        fig, ax = plt.subplots(figsize=(10, 4))
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
