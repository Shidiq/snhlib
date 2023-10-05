import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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
        self.loading_vector_ = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        self.loading_matrix_ = pd.DataFrame(data=self.loading_vector_, columns=self.pcnames)
        return

    def transform(self, X, y=None) -> None:
        score_vector = np.dot(X, self.loading_matrix_)
        drift_component = np.dot(score_vector, np.transpose(self.loading_vector_))
        Xcorr = X - drift_component
        return Xcorr
