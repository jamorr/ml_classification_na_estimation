import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import randomized_svd
from sklearn.impute import SimpleImputer

F32PREC = np.finfo(np.float32).eps


class SoftImputer(BaseEstimator, TransformerMixin):
    """

    Soft imputer based on
        https://github.com/BorisMuzellec/MissingDataOT/blob/master/softimpute.py
    Original Paper
        https://hastie.su.domains/Papers/mazumder10a.pdf

    """
    def __init__(self, copy:bool = True, verbose:bool = False,
        lmbd:float=0.1,
        max_iterations:int = 1000,
        stop_threshold:float = 1e-5) -> None:
        self.copy = copy
        self.verbose = verbose
        self.__fit = False
        self.max_iterations = max_iterations
        self.stop_threshold = stop_threshold
        self.lmbd = lmbd


    def _converged(self)->bool:
        """checks for convergence of inpute operation
        Converged when the root sum of squared difference in new and old imputed values ratio
        to the root sum of squared old imputed values is less than the stop threshold

        Returns:
            bool: True if impute has converged
        """
        old_missing_data_imputed_values = self.X[self.mask]
        new_missing_data_imputed_values = self.res[self.mask]
        rmse = np.sqrt(
            np.sum(
                (old_missing_data_imputed_values - new_missing_data_imputed_values) ** 2
                )
            )
        denom = np.sqrt(
            (old_missing_data_imputed_values ** 2).sum()
            )

        if denom == 0 or (denom < F32PREC and rmse > F32PREC):
            return False
        else:
            return (rmse / denom) < self.stop_threshold


    def fit_transform(
        self,
        X:np.ndarray, y=None
        )->np.ndarray|None:
        """
        Impute data using svd soft impute algorithm

        Args:
            data (np.ndarray): np array with missing values as np.nan
            self.lmbd (float, optional): Regularization parameter. Defaults to 0.1.
            self.max_iterations (int, optional): Defaults to 1000.
            self.stop_threshold (float, optional): convergence threshold. Defaults to 1e-5.

        Returns:
            np.ndarray|None: data with imputed values if copy == True
        """

        if self.copy:
            self.X = X.copy()
        else:
            self.X = X
        self.stop_threshold = self.stop_threshold
        self.mask = ~np.isnan(X)
        self.X = SimpleImputer().fit_transform(self.X)

        for _ in range(self.max_iterations):
            # use svd approximation for large matrices
            if self.X.shape[0]*self.X.shape[1] > 1e6:
                U, d, V = randomized_svd(self.X, n_components = np.minimum(200, self.X.shape[1]))
            else:
                U, d, V = np.linalg.svd(self.X, compute_uv = True, full_matrices=False)

            d_thresh = np.maximum(d - self.lmbd, 0)
            rank = (d_thresh > 0).sum()
            # select non zero singular values and vectors after threshold
            d_thresh = d_thresh[:rank]
            U_thresh = U[:, :rank]
            V_thresh = V[:rank, :]
            D_thresh = np.diag(d_thresh)
            self.res = U_thresh @ D_thresh @ V_thresh

            if self._converged():
                break

            self.X[~self.mask] = self.res[~self.mask]
        self.__fit = True
        if self.copy:
            return self.res

    def transform(self, X:np.ndarray) -> np.ndarray:
        if not self.__fit:
            raise AttributeError("Must fit before transform")
        mask = ~np.isnan(X)
        X = self.preprocessor.transform(X)
        coefficients, _, _, _ = np.linalg.lstsq(self.v, X.T, rcond=None)
        X[~mask] = np.dot(self.v, coefficients).T[~mask]
        return X

    def fit(
        self,
        X:np.ndarray, y=None
        )->np.ndarray|None:
        """
        Impute data using svd soft impute algorithm

        Args:
            data (np.ndarray): np array with missing values as np.nan
            self.lmbd (float, optional): Regularization parameter. Defaults to 0.1.
            self.max_iterations (int, optional): Defaults to 1000.
            self.stop_threshold (float, optional): convergence threshold. Defaults to 1e-5.

        Returns:
            np.ndarray|None: data with imputed values if copy == True
        """

        if self.copy:
            self.X = X.copy()
        else:
            self.X = X
        self.stop_threshold = self.stop_threshold
        self.mask = ~np.isnan(X)
        self.X = SimpleImputer().fit_transform(self.X)

        for _ in range(self.max_iterations):
            # use svd approximation for large matrices
            if self.X.shape[0]*self.X.shape[1] > 1e6:
                U, d, V = randomized_svd(self.X, n_components = np.minimum(200, self.X.shape[1]))
            else:
                U, d, V = np.linalg.svd(self.X, compute_uv = True, full_matrices=False)

            d_thresh = np.maximum(d - self.lmbd, 0)
            rank = (d_thresh > 0).sum()
            # select non zero singular values and vectors after threshold
            d_thresh = d_thresh[:rank]
            U_thresh = U[:, :rank]
            V_thresh = V[:rank, :]
            D_thresh = np.diag(d_thresh)
            self.res = U_thresh @ D_thresh @ V_thresh

            if self._converged():
                break

            self.X[~self.mask] = self.res[~self.mask]
        self.__fit = True
        del self.X
