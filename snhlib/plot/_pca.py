import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from snhlib.plot._confidence_ellipse import confidence_ellipse
from snhlib.plot._style import custom_style


class CalcPCA:
    """Analyze and plotting data with Principal Component Analysis"""

    def __init__(self, **kwargs) -> None:
        """__init__"""
        self.X = None
        self.y = None
        self.pc_ = None
        self.eig_ = None
        self.var_ = None
        self.pca_ = PCA()
        self.scaler = kwargs.get("scaler", StandardScaler)
        self.featurename = kwargs.get("featurename", None)
        self.showfliers = kwargs.get("showfliers", True)
        self.contamination = kwargs.get("contamination", 0.1)
        self.colors = kwargs.get("palette", None)
        self.markers = kwargs.get(
            "markers", ["o", "v", "s", "p", "P", "*", "h", "H", "X", "D", "+", "x", "d"]
        )

    def __repr__(self) -> str:
        return "Analyze and plotting data with principal component analysis"

    def fit(self, X, y):
        """fit
        fitting PCA

        Parameters
        ----------
        X : _type_
            value
        y : _type_
            list of id
        """
        self.X = X
        self.y = y

        if not self.showfliers:
            iso = IsolationForest(contamination=float(self.contamination))
            yhat = iso.fit_predict(self.X)
            mask = yhat != -1
            self.X, self.y = self.X[mask, :], self.y[mask]

        if self.scaler is not None:
            scaler = self.scaler()
            self.X = scaler.fit_transform(self.X)

        # fitting PCA
        self.pca_.fit(X=self.X)

        pc_score = self.pca_.transform(self.X)
        pc_name = [f"PC{i+1}" for i in range(pc_score.shape[1])]

        if self.featurename is None:
            self.featurename = [f"F{i+1}" for i in range(self.X.shape[1])]

        # Variance PC
        var_exp = np.round(self.pca_.explained_variance_ratio_ * 100, decimals=2)
        self.var_ = pd.DataFrame({"Var (%)": var_exp, "PC": pc_name})

        pc_score_df = pd.DataFrame(data=pc_score, columns=pc_name)
        val_y_df = pd.DataFrame(data=y, columns=["label"])
        self.pc_ = pd.concat([pc_score_df, val_y_df], axis=1)
        self.eig_ = pd.DataFrame(
            data=np.transpose(self.pca_.components_),
            columns=pc_name,
            index=self.featurename,
        )

    def scoreplot(self, **kwargs):
        """scoreplot
        score plot of PCA analysis

        Returns
        -------
        matplotlib.figure.Figure
        """
        PC = kwargs.get("PC", ["PC1", "PC2"])
        s = kwargs.get("size", 90)
        ellips = kwargs.get("ellips", True)
        ascending = kwargs.get("ascending", True)
        legend = kwargs.get("legend", True)
        loc = kwargs.get("loc", "best")

        if ascending:
            self.pc_ = self.pc_.sort_values(by=["label"], ascending=ascending)

        targets = list(self.pc_["label"].unique())

        if self.colors is None:
            colors = sns.color_palette("Paired", len(targets))
        else:
            colors = self.colors[: len(targets)]

        markers = self.markers[: len(targets)]

        xlabel = f'{PC[0]} ({float(self.var_.values[self.var_["PC"] == PC[0], 0])}%)'
        ylabel = f'{PC[1]} ({float(self.var_.values[self.var_["PC"] == PC[1], 0])}%)'

        fig, ax = custom_style()
        for target, color, mark in zip(targets, colors, markers):
            indicesToKeep = self.pc_["label"] == target
            x = self.pc_.loc[indicesToKeep, PC[0]]
            y = self.pc_.loc[indicesToKeep, PC[1]]
            ax.scatter(x, y, c=color, marker=mark, s=s, label=str(target))

            if ellips:
                confidence_ellipse(x, y, ax, edgecolor=color)

        ax.set_xlabel(xlabel, fontsize=28, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=28, fontweight="bold")

        if legend:
            if loc == 5:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            else:
                ax.legend(loc=loc)

        return fig, ax

    def screenplot(self, **kwargs):
        """screenplot
        Screen plot of PCA Analysis

        Returns
        -------
        matplotlib.figure.Figure
        """
        lim = kwargs.get("PC", None)

        if lim is None:
            data_ = self.var_
        else:
            data_ = self.var_.loc[:lim, :]

        fig = custom_style(single_ax=False)
        sns.pointplot(x="PC", y="Var (%)", data=data_)
        plt.xticks(rotation="vertical")
        plt.xlabel("Principal Component")
        plt.ylabel("Percentage of Variance (%)")
        return fig

    def loadingplot(self, **kwargs):
        """loadingplot
        Loading plot of PCA Analysis

        Returns
        -------
        matplotlib.figure.Figure
        """
        lim = kwargs.get("alim", 1.1)
        circle = kwargs.get("circle", 2)

        PC = ["PC1", "PC2"]
        xlabs = f'{PC[0]} ({float(self.var_.values[self.var_["PC"] == PC[0], 0])}%)'
        ylabs = f'{PC[1]} ({float(self.var_.values[self.var_["PC"] == PC[1], 0])}%)'

        fig, ax = custom_style()
        for i in range(0, self.pca_.components_.shape[1]):
            ax.arrow(
                0,
                0,
                self.pca_.components_[0, i],
                self.pca_.components_[1, i],
                head_width=0.05,
                head_length=0.05,
            )
            plt.text(
                self.pca_.components_[0, i] + 0.05,
                self.pca_.components_[1, i] + 0.05,
                self.featurename[i],
                size=18,
            )

        an = np.linspace(0, circle * np.pi, 100)
        plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
        plt.axis("equal")
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_xlabel(xlabs, fontsize=28, fontweight="bold")
        ax.set_ylabel(ylabs, fontsize=28, fontweight="bold")
        plt.axhline(y=0.0, color="b", linestyle="--")
        plt.axvline(x=0.0, color="b", linestyle="--")
        return fig

    def getbestfeature(self, PC=0, n=3):
        """getbestfeature
        Get n-best features of PCA Analysis

        Parameters
        ----------
        PC : int, optional
            PC number, by default 0
        n : int, optional
            number of n-best features, by default 3
        """
        loading_score = pd.Series(self.pca_.components_[PC], index=self.featurename)
        sorted_loading_score = loading_score.abs().sort_values(ascending=False)
        top_score = sorted_loading_score[0:n].index.values
        print(loading_score[top_score])
