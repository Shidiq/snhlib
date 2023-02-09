import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from snhlib.plot._confidence_ellipse import confidence_ellipse
from snhlib.plot._style import custom_style


class CalcLDA:
    """
    ============================
    Linear discriminant analysis
    ============================

    CalcLDA(round_, scaler, cv)

    Methods:

    - fit(x, y)
    - getvarkd()
    - getscore()
    - plotlda(adj_left, adj_bottom, acending)

    """

    def __init__(self, **kwargs):
        self.X = None
        self.Xval = None
        self.y = None
        self.yval = None
        self.ld_val_ = None
        self.dual = None

        self.var_ = pd.DataFrame()
        self.ld_ = pd.DataFrame()
        self.lda_ = LinearDiscriminantAnalysis()
        self.scaler = kwargs.get("scaler", StandardScaler())
        self.colors = kwargs.get("colors", None)
        self.showfliers = kwargs.get("showfliers", True)
        self.contamination = kwargs.get("contamination", 0.1)
        self.markers = kwargs.get(
            "markers", ["o", "v", "s", "p", "P", "*", "h", "H", "X", "D", "+", "x", "d"]
        )
        self.cv = kwargs.get("cv", 10)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"{self.lda_!r})"

    def fit(self, *arrays):
        if len(arrays) == 2:
            self.X = arrays[0]
            self.y = arrays[1]
            self.dual = False
        else:
            self.X = arrays[0]
            self.Xval = arrays[1]
            self.y = arrays[2]
            self.yval = arrays[3]
            self.ld_val_ = None
            self.dual = True

        if not self.showfliers:
            iso = IsolationForest(contamination=float(self.contamination))
            yhat = iso.fit_predict(self.X)
            mask = yhat != -1
            self.X, self.y = self.X[mask, :], self.y[mask]

        scaler = self.scaler
        X = scaler.fit_transform(self.X)
        self.lda_.fit(X, self.y)
        ldax = self.lda_.transform(X)

        ldname = [f"LD{i + 1}" for i in range(ldax.shape[1])]
        self.ld_ = pd.DataFrame(ldax, columns=ldname)
        Y = pd.DataFrame(data=self.y, columns=["label"])
        self.ld_ = pd.concat([self.ld_, Y], axis=1)

        tot = sum(self.lda_.explained_variance_ratio_)
        var_exp = [
            round((i / tot) * 100, 2)
            for i in sorted(self.lda_.explained_variance_ratio_, reverse=True)
        ]
        self.var_ = pd.DataFrame({"Var (%)": var_exp, "LD": ldname})

        if self.dual:
            Xval = scaler.transform(self.Xval)
            ldax = self.lda_.transform(Xval)
            self.ld_val_ = pd.DataFrame(ldax, columns=ldname)
            Y = pd.DataFrame(data=self.yval, columns=["label"])
            self.ld_val_ = pd.concat([self.ld_val_, Y], axis=1)

            ldaDF1 = pd.concat(
                [
                    self.ld_,
                    pd.DataFrame(data=self.ld_["label"].values, columns=["Class"]),
                ],
                axis=1,
            )
            ldaDF1["Class"] = "Training"

            ldaDF2 = pd.concat(
                [
                    self.ld_val_,
                    pd.DataFrame(data=self.ld_val_["label"].values, columns=["Class"]),
                ],
                axis=1,
            )
            ldaDF2["Class"] = "Testing"

            self.ld_ = pd.concat([ldaDF1, ldaDF2], axis=0)

    def scoreplot(self, **kwargs):
        elip = kwargs.get("ellipse", True)
        ascending = kwargs.get("ascending", True)
        legend = kwargs.get("legend", True)
        loc = kwargs.get("loc", "best")

        self.ld_ = self.ld_.sort_values(by=["label"], ascending=ascending)
        nlabel = np.unique(self.y)

        if len(nlabel) < 3:
            fig, ax = custom_style()
            s = kwargs.get("size", 10)

            if self.dual:
                self.ld_val_ = self.ld_val_.sort_values(by=["label"], ascending=ascending)
                ax = sns.stripplot(x="label", y="LD1", color="k", size=s, data=self.ld_)
                ax = sns.stripplot(
                    x="label",
                    y="LD1",
                    marker="^",
                    color="red",
                    size=s,
                    data=self.ld_val_,
                )
            else:
                ax = sns.stripplot(x="label", y="LD1", size=s, data=self.ld_)

            ax.set_xlabel("Classes")
            ax = plt.axhline(y=0, linewidth=1.5, color="black", linestyle="--")
            return fig
        else:
            targets = list(self.ld_["label"].unique())

            s = kwargs.get("size", 90)
            if len(targets) > 10:
                raise ValueError(str(targets))

            if self.colors is None:
                colors = sns.color_palette("Paired", len(targets))
            else:
                colors = self.colors[: len(targets)]

            markers = self.markers[: len(targets)]

            xlabs = f"LD1 ({self.var_.values[0, 0]}%)"
            ylabs = f"LD2 ({self.var_.values[1, 0]}%)"

            fig, ax = custom_style()
            for target, color, mark in zip(targets, colors, markers):
                indicesToKeep = self.ld_["label"] == target
                x = self.ld_.loc[indicesToKeep, "LD1"]
                y = self.ld_.loc[indicesToKeep, "LD2"]
                ax.scatter(x, y, c=color, marker=mark, s=s, label=target)

                if elip:
                    confidence_ellipse(x, y, ax, edgecolor=color)

            if self.dual:
                self.ld_val_ = self.ld_val_.sort_values(by=["label"], ascending=ascending)

                for target, color, mark in zip(targets, colors, markers):
                    indicesToKeep = self.ld_val_["label"] == target
                    x = self.ld_val_.loc[indicesToKeep, "LD1"]
                    y = self.ld_val_.loc[indicesToKeep, "LD2"]
                    ax.scatter(
                        x,
                        y,
                        marker=mark,
                        s=s,
                        facecolors="none",
                        edgecolors=color,
                        label=f"{target} - test",
                    )

            if legend:
                ax.legend(loc=loc)

            ax.set_xlabel(xlabs)
            ax.set_ylabel(ylabs)

            return fig
