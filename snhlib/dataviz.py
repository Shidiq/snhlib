import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from snhlib.image import Style
from snhlib.utils import confidence_ellipse
import numpy as np


class CalcPCA:
    """
    ============================
    Principal component analysis
    ============================

    CalcPCA(round_, featurename, scaler)

    Methods:

    - fit(x, y)
    - getvarpc()
    - getcomponents()
    - getbestfeature()
    - plotpc(PC, adj_left, adj_bottom, acending)
    - screenplot(adj_left, adj_Bottom)

    """

    def __init__(self, **options):
        self.x = None
        self.y = None
        self.vardf = pd.DataFrame()
        self.pcadf = pd.DataFrame()
        self.eigpc = pd.DataFrame()
        self.round_ = options.get('round_', 1)
        self.featurename = options.get('featurename', None)
        self.scaler = options.get('scaler', StandardScaler())
        self.colors = options.get(
            'colors', ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'C0', 'C1', 'C2'])
        self.markers = options.get(
            'markers', ["o", "v", "s", "p", "P", "*", "h", "H", "X", "D"])
        self.pca = PCA()

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'{self.pca!r})'
        )

    def fit(self, x, y):
        self.x = x
        self.y = y

        if self.scaler is not None:
            scaler = self.scaler
            self.x = scaler.fit_transform(self.x)

        self.pca = PCA()
        self.pca.fit(self.x)
        pcscore = self.pca.transform(self.x)
        pcname = [f'PC{i + 1}' for i in range(pcscore.shape[1])]
        # pcname = [f'PC{i + 1}' for i in range(self.x.shape[1])]
        if self.featurename is None:
            self.featurename = [
                f'Feature{i + 1}' for i in range(self.x.shape[1])]
        # var_exp = [round(i * 100, self.round_) for i in sorted(self.pca.explained_variance_ratio_, reverse=True)]
        var_exp = np.round(
            self.pca.explained_variance_ratio_ * 100, decimals=self.round_)
        self.vardf = pd.DataFrame({'Var (%)': var_exp, 'PC': pcname})
        # pcscore = self.pca.transform(self.x)
        pcaDF = pd.DataFrame(data=pcscore, columns=pcname)
        Y = pd.DataFrame(data=self.y, columns=['label'])
        self.pcadf = pd.concat([pcaDF, Y], axis=1)
        self.eigpc = pd.DataFrame(data=np.transpose(
            self.pca.components_), columns=pcname, index=self.featurename)
        return self.pca

    def getvarpc(self):
        return self.pcadf, self.vardf, self.eigpc

    def getcomponents(self):
        loading_score = pd.DataFrame(
            data=self.pca.components_, columns=[self.featurename])
        return loading_score

    def getbestfeature(self, PC=0, n=3):
        loading_score = pd.Series(
            self.pca.components_[PC], index=self.featurename)
        sorted_loading_score = loading_score.abs().sort_values(ascending=False)
        top_score = sorted_loading_score[0:n].index.values
        print(loading_score[top_score])

    def plotpc(self, **options):
        PC = options.get('PC', ['PC1', 'PC2'])
        # a = options.get('adj_left', 0.1)
        # b = options.get('adj_bottom', 0.15)
        s = options.get('size', 90)
        elip = options.get('ellipse', True)
        ascending = options.get('ascending', True)

        self.pcadf = self.pcadf.sort_values(by=['label'], ascending=ascending)

        targets = list(self.pcadf['label'].unique())

        if len(targets) > 10:
            raise ValueError(str(targets))

        colors = self.colors[:len(targets)]
        markers = self.markers[:len(targets)]

        xlabs = f'{PC[0]} ({float(self.vardf.values[self.vardf["PC"] == PC[0], 0])}%)'
        ylabs = f'{PC[1]} ({float(self.vardf.values[self.vardf["PC"] == PC[1], 0])}%)'

        fig, ax = Style().paper()
        for target, color, mark in zip(targets, colors, markers):
            indicesToKeep = self.pcadf['label'] == target
            x = self.pcadf.loc[indicesToKeep, PC[0]]
            y = self.pcadf.loc[indicesToKeep, PC[1]]
            ax.scatter(x, y, c=color, marker=mark, s=s, label=str(target))
            if elip:
                confidence_ellipse(x, y, ax, edgecolor=color)

        plt.xlabel(xlabs)
        plt.ylabel(ylabs)
        plt.legend(targets)
        return fig

    def screenplot(self, **options):
        # a = options.get('adj_left', 0.1)
        # b = options.get('adj_bottom', 0.2)
        lim = options.get('PC', None)

        if lim is None:
            data_ = self.vardf
        else:
            data_ = self.vardf.loc[:lim, :]

        fig, _ = Style().paper()
        plt.bar(x='PC', height='Var (%)', data=data_)
        plt.xticks(rotation='vertical')
        plt.xlabel('Principal Component')
        plt.ylabel('Percentage of Variance')
        return fig
