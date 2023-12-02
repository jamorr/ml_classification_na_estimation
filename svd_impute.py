import scipy
import numpy as np
from sklearnex import patch_sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from kneed import KneeLocator

patch_sklearn()


class SVDImpute():
    def __init__(self) -> None:
        pass

    def _find_optimal_k(self, X):
        n = min(X.shape)
        model = PCA(n_components=n)
        model.fit_transform(X)

        pca_evr = model.explained_variance_ratio_
        pca_evr = np.concatenate((np.zeros(1), pca_evr))
        indx, cum_evr  = np.arange(n+1), np.cumsum(pca_evr)

        kneedle = KneeLocator(indx, cum_evr, direction='increasing', curve='concave', S=1.0, interp_method='polynomial', online=True)
        return kneedle.knee

    def fit_transform(self, X):
        X = SimpleImputer().fit_transform(X)
        k = self._find_optimal_k(X)
        u, s, vt = scipy.sparse.linalg.svds(X, k=k)
        return u @ np.diag(s) @ vt