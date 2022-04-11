from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import BatchNormalization, Dense, Input, LeakyReLU
from tensorflow.keras.models import Model

from snhlib.utils import ProgressBar


class AutoEncoder:
    def __init__(
        self,
        n_target: int,
        test_size=0.2,
        epochs=200,
        batch_size=16,
        random_state=None,
        verbose=2,
    ) -> None:
        self.n_bottleneck = n_target
        self.test_size = test_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.history_ = None
        self.encoder_ = None
        return

    def train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        return X_train, X_test, y_train, y_test

    def fit(self, X, y) -> None:
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)

        t = MinMaxScaler()
        t.fit(X_train)
        X_train = t.transform(X_train)
        X_test = t.transform(X_test)

        # define encoder
        n_inputs = X.shape[1]
        visible = Input(shape=(n_inputs,))

        # encoder level 1
        e = Dense(n_inputs * 2)(visible)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)

        # encoder level 2
        e = Dense(n_inputs)(e)
        e = BatchNormalization()(e)
        e = LeakyReLU()(e)

        # bottleneck
        n_bottleneck = self.n_bottleneck
        bottleneck = Dense(n_bottleneck)(e)

        # define decoder level 1
        d = Dense(n_inputs)(bottleneck)
        d = BatchNormalization()(d)
        d = LeakyReLU()(d)

        # output layer
        output = Dense(n_inputs, activation="linear")(d)

        # define autoencoder model
        model = Model(inputs=visible, outputs=output)

        # compile autoencoder model
        model.compile(optimizer="adam", loss="mse")

        self.history_ = model.fit(
            X_train,
            X_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=(X_test, X_test),
        )

        self.encoder_ = Model(inputs=visible, outputs=bottleneck)
        return

    @property
    def plot_history_(self):
        plt.plot(self.history_.history["loss"], label="train")
        plt.plot(self.history_.history["val_loss"], label="test")
        plt.legend()
        plt.show()


class HybridHAC:
    def __init__(
        self, colnames, n_repeats=100, step=0.01, random_state=None, linewidth=3
    ) -> None:
        self.colnames = colnames
        self.n_repeats = n_repeats
        self.step = step
        self.random_state = random_state
        self.linewidth = linewidth

        self.dist_linkage_ = None
        self.df_clust_ = None
        self.list_n_clust_ = None
        self.col_sorted_ = None
        self.importances_ = None
        return

    def plot_clust(self, X):
        dist_linkage_ = self.get_dist_lingkage(X)
        fig, ax = self.paper()
        matplotlib.rcParams["lines.linewidth"] = self.linewidth
        _ = hierarchy.dendrogram(
            dist_linkage_, labels=self.colnames, ax=ax, leaf_rotation=90
        )
        plt.xticks(fontsize=28)
        ax.set_xlabel("Features")
        ax.set_ylabel("Distance")
        return fig

    def fit(self, model, X, y):
        progress = ProgressBar(5, fmt=ProgressBar.FULL)

        self.dist_linkage_ = self.get_dist_lingkage(X)
        progress.current += 1
        progress()

        self.df_clust_ = self.get_data_cluster()
        progress.current += 1
        progress()

        self.list_n_clust_ = self.get_n_cluster()
        progress.current += 1
        progress()

        self.col_sorted_ = self.get_imp_idx(model, X, y)
        progress.current += 1
        progress()

        self.importances_ = self.get_imp_clust()
        progress.current += 1
        progress()
        progress.done()
        return

    @staticmethod
    def get_dist_lingkage(X):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        corr = spearmanr(Xs).correlation
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)
        distance_matrix = 1 - np.abs(corr)
        dist_linkage = hierarchy.ward(squareform(distance_matrix))
        return dist_linkage

    def get_data_cluster(self):
        dendro = hierarchy.dendrogram(
            self.dist_linkage_, labels=self.colnames, no_plot=True
        )
        dd = dendro["dcoord"]
        dmax = max([max(dd[i]) for i in range(len(dd))])
        dmin = min(min(dendro["dcoord"]))
        list_d = np.arange(dmin, dmax, step=self.step)

        dict_clust = {"d": [], "c": []}
        df_clust = pd.DataFrame()
        fclus = [0 for i in range(len(self.colnames))]

        for i, d in enumerate(list_d):
            fclus_new = fcluster(self.dist_linkage_, t=d, criterion="distance")
            if any(fclus != fclus_new):
                fclus = fclus_new
                fclus_ = [[f] for f in fclus]
                res = dict(zip(self.colnames, fclus_))
                dict_clust["d"].append(d)
                dict_clust["c"].append(res)
                res = pd.DataFrame.from_dict(res)
                res.index = [round(d, 3)]
                df_clust = pd.concat([df_clust, res], axis=0)
        df_clust.index.names = ["distance"]
        return df_clust

    def get_n_cluster(self):
        list_n_clust = []
        dfT = self.df_clust_.T
        for dist in list(dfT):
            d_clust = dfT[dist]
            n_clust = {key: [] for key in np.unique(d_clust)}
            for i, item in enumerate(d_clust):
                n_clust[item].append(d_clust.index[i])
            list_n_clust.append(n_clust)
        return list_n_clust

    def get_imp_idx(self, estimator, X, y):
        permut = permutation_importance(
            estimator, X, y, n_repeats=self.n_repeats, random_state=self.random_state
        )
        perm_sorted_idx = permut.importances_mean.argsort()
        col_sorted = [self.colnames[i] for i in perm_sorted_idx]
        return col_sorted

    @staticmethod
    def get_imp_branch(list_select, col_sorted):
        idx = [col_sorted.index(i) for i in list_select]
        out = list_select[idx.index(max(idx))]
        return out

    def get_imp_clust(self):
        list_d = self.df_clust_.index
        result = {key: [] for key in list_d}
        for d, n_clust in enumerate(self.list_n_clust_):
            for clust in n_clust.keys():
                out = self.get_imp_branch(n_clust[clust], self.col_sorted_)
                result[list_d[d]].append(out)
        return result

    @staticmethod
    def paper(loc="best", classic=True, figsize=[10.72, 8.205]):
        if classic:
            plt.style.use("classic")
        params = {
            "axes.formatter.useoffset": False,
            "font.family": "sans-serif",
            "font.sans-serif": "Arial",
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "axes.labelsize": 28,
            "axes.labelweight": "bold",
            "axes.titlesize": 28,
            "axes.titleweight": "bold",
            "figure.dpi": 300,
            "figure.figsize": figsize,
            "legend.loc": loc,
            "legend.fontsize": 24,
            "legend.fancybox": True,
            "mathtext.fontset": "custom",
            "mathtext.default": "regular",
            "figure.autolayout": True,
            "patch.edgecolor": "#000000",
            "text.color": "#000000",
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
        matplotlib.rcParams.update(params)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.patch.set_facecolor("xkcd:white")
        return fig, ax


class CorrectionComponent:
    def __init__(self, n_component=1) -> None:
        self.n_component = n_component
        self.pca = None
        self.loading_matrix_ = None
        self.loading_vector_ = None
        self.pcnames = [f"PC{i+1}" for i in range(self.n_component)]

    def fit(self, X, y=None) -> None:

        self.pca = PCA(n_components=self.n_component)
        self.pca.fit(X)
        self.loading_vector_ = self.pca.components_.T * np.sqrt(
            self.pca.explained_variance_
        )
        self.loading_matrix_ = pd.DataFrame(
            data=self.loading_vector_, columns=self.pcnames
        )
        return

    def transform(self, X, y=None) -> None:
        score_vector = np.dot(X, self.loading_matrix_)
        drift_component = np.dot(
            score_vector, np.transpose(self.loading_vector_))
        Xcorr = X - drift_component
        return Xcorr
