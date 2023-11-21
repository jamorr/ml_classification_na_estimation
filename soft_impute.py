import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.impute import SimpleImputer

F32PREC = np.finfo(np.float32).eps


class SoftImputer():
    """

    Soft imputer based on
        https://github.com/BorisMuzellec/MissingDataOT/blob/master/softimpute.py
    Original Paper
        https://hastie.su.domains/Papers/mazumder10a.pdf

    """
    def __init__(self, copy:bool = True, verbose:bool = False) -> None:
        self.copy = copy
        self.verbose = verbose

    def _converged(self)->bool:
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
        denom = np.sqrt(
            (old_missing_data_imputed_values ** 2).sum()
            )

        if denom == 0 or (denom < F32PREC and rmse > F32PREC):
            return False
        else:
            return (rmse / denom) < self.stop_threshold


    def fit_transform(
        self,
        data:np.ndarray,
        lmbd:float=0.1,
        max_iterations:int = 1000,
        stop_threshold:float = 1e-5)->np.ndarray|None:
        """
        Impute data using svd soft impute algorithm

        Args:
            data (np.ndarray): np array with missing values as np.nan
            lmbd (float, optional): Regularization parameter. Defaults to 0.1.
            max_iterations (int, optional): Defaults to 1000.
            stop_threshold (float, optional): convergence threshold. Defaults to 1e-5.

        Returns:
            np.ndarray|None: data with imputed values if copy == True
        """

        if self.copy:
            self.data = data.copy()
        else:
            self.data = data
        self.stop_threshold = stop_threshold
        self.mask = ~np.isnan(data)
        self.data = SimpleImputer().fit_transform(self.data)

        for _ in range(max_iterations):
            # use svd approximation for large matrices
            if self.data.shape[0]*self.data.shape[1] > 1e6:
                U, d, V = randomized_svd(self.data, n_components = np.minimum(200, self.data.shape[1]))
            else:
                U, d, V = np.linalg.svd(self.data, compute_uv = True, full_matrices=False)

            d_thresh = np.maximum(d - lmbd, 0)
            rank = (d_thresh > 0).sum()
            # select non zero singular values and vectors after threshold
            d_thresh = d_thresh[:rank]
            U_thresh = U[:, :rank]
            V_thresh = V[:rank, :]
            D_thresh = np.diag(d_thresh)
            self.res = U_thresh @ D_thresh @ V_thresh

            if self._converged():
                break

            self.data[~self.mask] = self.res[~self.mask]

        if self.copy:
            return self.res