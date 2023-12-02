import scipy
import numpy as np
from sklearnex import patch_sklearn
from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import randomized_svd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from kneed import KneeLocator

patch_sklearn()

F32PREC = np.finfo(np.float32).eps


class SVDImputer(BaseEstimator, TransformerMixin):
    def __init__(self,max_iterations: int = 1000, stop_threshold: float = 1e-5 ) -> None:
        self.__fit = False
        self.stop_threshold = stop_threshold
        self.max_iterations = max_iterations

    def _converged(self) -> bool:
        """checks for convergence of inpute operation
        Converged when the root sum of squared difference in new and old imputed values ratio
        to the root sum of squared old imputed values is less than the stop threshold

        Returns:
            bool: True if impute has converged
        """
        old_missing_data_imputed_values = self.data[self.mask]
        new_missing_data_imputed_values = self.res[self.mask]
        rmse = np.sqrt(
            np.sum(
                (old_missing_data_imputed_values - new_missing_data_imputed_values) ** 2
            )
        )
        denom = np.sqrt((old_missing_data_imputed_values**2).sum())

        if denom == 0 or (denom < F32PREC and rmse > F32PREC):
            return False
        else:
            return (rmse / denom) < self.stop_threshold

    def _find_optimal_k(self, X):
        n = min(X.shape)
        model = PCA(n_components=n)
        model.fit_transform(X)

        pca_evr = model.explained_variance_ratio_
        pca_evr = np.concatenate((np.zeros(1), pca_evr))
        indx, cum_evr = np.arange(n + 1), np.cumsum(pca_evr)

        kneedle = KneeLocator(
            indx,
            cum_evr,
            direction="increasing",
            curve="concave",
            S=1.0,
            interp_method="polynomial",
            online=True,
        )
        return kneedle.knee

    def fit_transform(
        self, X, y=None
    ) -> np.ndarray | None:

        self.mask = ~np.isnan(X)
        self.preprocessor = SimpleImputer()
        self.data = self.preprocessor.fit_transform(X)
        k = self._find_optimal_k(self.data)

        for _ in range(self.max_iterations):

            if self.data.shape[0] * self.data.shape[1] > 1e6:
                u, s, vt = randomized_svd(self.data, n_components=k)
            else:
                # u, s, vt = np.linalg.svd(self.data, compute_uv = True, full_matrices=False)
                u, s, vt = scipy.sparse.linalg.svds(self.data, k=k)
            self.u = u
            self.s = np.diag(s)
            self.v = vt.T
            self.res = self.u @ self.s @ vt

            if self._converged():
                break
            self.data[~self.mask] = self.res[~self.mask]
        self.__fit = True
        return self.data

    def transform(self, X:np.ndarray)->np.ndarray:
        if not self.__fit:
            raise AttributeError("Must fit before transform")
        mask = ~np.isnan(X)
        X = self.preprocessor.transform(X)
        coefficients, _, _, _ = np.linalg.lstsq(self.v, X.T, rcond=None)
        X[~mask] = np.dot(self.v, coefficients).T[~mask]
        return X

    def fit(
        self, X, y=None
    ) -> np.ndarray | None:

        self.mask = ~np.isnan(X)
        self.preprocessor = SimpleImputer()
        self.data = self.preprocessor.fit_transform(X)
        k = self._find_optimal_k(self.data)

        for _ in range(self.max_iterations):
            if self.data.shape[0] * self.data.shape[1] > 1e6:
                u, s, vt = randomized_svd(self.data, n_components=k)
            else:
                # u, s, vt = np.linalg.svd(self.data, compute_uv = True, full_matrices=False)
                u, s, vt = scipy.sparse.linalg.svds(self.data, k=k)
            self.u = u
            self.s = np.diag(s)
            self.v = vt.T
            self.res = self.u @ self.s @ vt

            if self._converged():
                break
            self.data[~self.mask] = self.res[~self.mask]
        del self.data
        self.__fit = True


if __name__ == "__main__":
    import utils

    data = utils.read_missing("./missing/MissingData1.txt")
    model = SVDImputer()
    ft_data = model.fit_transform(data)
    print(data.shape)
    t_data = model.test_transform(data)
    print(print(np.linalg.norm(t_data - ft_data, ord="fro")))
