from functools import reduce
from itertools import chain
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal  # for generating pdf
from sklearn.datasets import make_spd_matrix
from sklearn.impute import SimpleImputer

rng = np.random.default_rng()

def gen_data(
        num_dims:int = 2,
        num_clusters:int = 6,
        mean_min: int = -10,
        mean_max: int = 10,
        cluster_size_min: int = 20,
        cluster_size_max:int = 1000
    ):
        """
        num_dims:int = 2
            number of dimensions of data (columns)
        num_clusters:int = 6
            number of clusters in data
        mean_min: int = -10
            lower bound for means
        mean_max: int = 10
            upper bound for means
        cluster_size_min: int = 20
            minimum number of points in a cluster
        cluster_size_max:int = 1000
            maximum number of points in a cluster

        """

        print("Generating data...")
        covars = np.array([make_spd_matrix(num_dims) for _ in range(num_clusters)])
        means = rng.uniform(low=mean_min, high=mean_max, size=(num_clusters, num_dims))

        clusters = []
        for m, cov in zip(means, covars):
            clusters.append(
                np.random.multivariate_normal(
                    m, cov,
                    size=(rng.integers(cluster_size_min,cluster_size_max),)
                    )
                )
        gt_labels = np.array(
            list(
                chain.from_iterable(
                    [[[i]]*len(clusters[i]) for i in range(num_clusters)]
                    )
                )
            )
        data = np.concatenate([c for c in clusters], axis=0)
        print("Data generated successfully.")
        return data, gt_labels

class ExpectationMaximization:

    def __init__(self) -> None:
        # self.true_means
        # self.true_cov

        self.__fitted = False
        self.__predicted = False

    def _expectation_step(self):
        expect_z = np.ndarray((len(self.data),self.n_dist), np.float64)
        if self.estimate_proportion:
            for i, (mean, variance) in enumerate(zip(self.means, self.cov)):
                expect_z[:, i] = self.proportions[i] * multivariate_normal.pdf(self.data, mean, variance, allow_singular=True)
        else:
            for i, (mean, variance) in enumerate(zip(self.means, self.cov)):
                expect_z[:, i] = multivariate_normal.pdf(self.data, mean, variance, allow_singular=True)
        # likelihood = expect_z
        # smoothing_factor = 1e-20
        expect_z = (((expect_z.T + self.smoothing_factor) / (expect_z.sum(1) + len(expect_z) * self.smoothing_factor))).T
        self.likelihood = np.sum(np.log(expect_z.T)).sum()/len(self.data)
        self.pred_labels = expect_z

    def _maximization_step(self):
        self._expectation_step()
        # sum of weights for each hidden variable
        # weighted sum of data over sum of weights for each hidden variable
        try:
            self.means: np.ndarray = np.array(
                [
                    np.average(
                        self.data, axis=0, weights=e
                    )
                    for e in self.pred_labels.T
                ]
            )
        except ZeroDivisionError as e:
            print(self.pred_labels)
            raise e
        self.cov:np.ndarray = np.array(
            [
                np.cov(self.data.T, ddof = 0, aweights=e) for e in self.pred_labels.T
            ]
        )
        self.proportions: np.ndarray = self.pred_labels.sum(0) / len(self.data)

    def fit(self, data:Optional[np.ndarray]=None, labels:Optional[np.ndarray]=None, **kwargs):
        no_data = data is None
        no_labels = labels is None

        if no_data and no_labels:
            self.data, self.gt_labels = gen_data(**kwargs)
        elif no_data:
            raise ValueError("Must include both data")
        else:
            self.data = data

        if not no_labels:
            self.gt_labels = labels
        else:
            self.gt_labels = None

        self.__fitted = True
        return self

    def predict(
        self,
        number_of_distributions: int,
        stop: float = 1e-10,
        iterations:int = 1000,
        smoothing_factor: float = 1e-20,
        estimate_proportion:bool=True
    ):

        if not self.__fitted:
            raise AttributeError("No data has been fitted")

        self.estimate_proportion = estimate_proportion
        self.n_dist = number_of_distributions
        self.smoothing_factor = smoothing_factor
        self.means = rng.choice(self.data, number_of_distributions)
        # self.means = (
        #     rng.random((number_of_distributions, self.data.shape[1]), np.float64) + self.data.mean()
        # )
        # means = data[:number_of_distributions, :]
        self.cov = rng.random(
            (number_of_distributions, self.data.shape[1]), np.float64
        )
        self.cov = np.array([np.outer(v,v) for v in self.cov])
        for _ in range(number_of_distributions-1):
            vv = rng.random(
            (number_of_distributions, self.data.shape[1]), np.float64)
            self.cov += np.array([np.outer(v,v) for v in vv])

        self.proportions = (
            np.ones(number_of_distributions) * (1 / number_of_distributions) * len(self.data)
        )

        self.means.sort()
        last = 0
        k = float("inf")
        # stop when there is less than stop=1e-7 distance between new and old
        # in means and variance
        i = 0
        self.expect = np.ndarray(1)
        while k > stop and i < iterations:
            # get new mean/var
            self._maximization_step()
            # self.means, self.cov, self.proportions, self.expect, self.likelihood = new_hyp_mutli_var(self.data, self.means, self.cov, proportions, estimate_proportion)
            k, last = np.absolute(self.likelihood - last), self.likelihood
            i += 1
        print(f"Stopped after {i} iterations, with change in likelihood {k}")
        self.__predicted = True
        return self

    def plot_2d(self):
        if not self.__fitted:
            raise AttributeError("Must have data attribute to plot")
        if self.data.shape[1] != 2:
            raise ValueError("only able to plot 2d data")

        X1 = np.linspace(self.data[:,0].min()-1, self.data[:,0].max()+1, 250)
        X2 = np.linspace(self.data[:,1].min()-1, self.data[:,1].max()+1, 250)
        X, Y = np.meshgrid(X1, X2)


        pos = np.empty(X.shape + (2,))                # a new array of given shape and type, without initializing entries
        pos[:, :, 0], pos[:, :, 1] = X, Y

        # plt.figure(figsize=(10,10))
        # creating the figure and assigning the size
        if isinstance(self.gt_labels, np.ndarray) and self.gt_labels.any():
            groups = pd.DataFrame(np.concatenate((self.data, self.gt_labels), axis=1))
            axes = groups.plot.scatter(x=0, y=1, c=2, alpha = 0.5)
        else:
            axes = plt.scatter(self.data[:,0], self.data[:,1], marker='o', alpha = 0.5) # type: ignore

        if self.__predicted:
            distributions = []
            for m, c in zip(self.means, self.cov):
                distributions.append(multivariate_normal(m, c, allow_singular=True))
            colors = ['r','g','b','k','c', 'm']
            for dist, col in zip(distributions, colors):
                plt.contour(X, Y, dist.pdf(pos), colors=col ,alpha = 0.5)
        plt.axis('equal')                                                                  # making both the axis equal
        plt.xlabel('X-Axis', fontsize=16)                                                  # X-Axis
        plt.ylabel('Y-Axis', fontsize=16)                                                  # Y-Axis
        plt.grid()                                                                         # displaying gridlines
        plt.show()
        return axes


class EMImputer:

    def __init__(self, max_iter=100) -> None:
        # self.true_means
        # self.true_cov

        self.__fitted = False
        self.__predicted = False
        self.max_iter = max_iter

    # def fit_transform(
    #     self,
    #     data:np.ndarray,
    #     stop: float = 1e-7,
    #     iterations:int = 100,
    #     # smoothing_factor: float = 1e-20,
    #     # estimate_proportion:bool=True,
    #     # **kwargs
    #     ):
    #     """
    #     Code heavily based on : https://joon3216.github.io/research_materials/2019/em_imputation_python.html

    #     Args:
    #         data (Optional[np.ndarray], optional): Data with missing values. Defaults to None.
    #         stop (float, optional): Change in values to stop at. Defaults to 1e-7.
    #         iterations (int, optional): Max number of iterations. Defaults to 1000.

    #     Raises:
    #         ValueError: missing data

    #     Returns:
    #         _type_: _description_
    #     """
    #     no_data = data is None

    #     if no_data:
    #         raise ValueError("Must include data")
    #     self.data = data


    #     num_rows, num_columns = self.data.shape
    #     mask_missing = ~np.isnan(self.data)

    #     # Collect M_i and O_i's
    #     one_to_num_columns = np.arange(1, num_columns + 1, step = 1)
    #     rows_w_missing = one_to_num_columns * (~mask_missing) - 1
    #     rows_w_observed = one_to_num_columns * mask_missing - 1

    #     # Generate Mu_0 and Sigma_0
    #     Mu = np.nanmean(self.data, axis = 0)

    #     # observed_rows = np.where(np.isnan(sum(self.data.T)) == False)[0]
    #     observed_rows = np.where(~np.isnan(sum(self.data.T)))[0]
    #     if observed_rows.size > 0:
    #         cov_matrix = np.cov(self.data[observed_rows, ].T)
    #     else:
    #         cov_matrix = np.diag(np.nanvar(self.data, axis = 0))

    #     if np.isnan(cov_matrix).any():
    #         cov_matrix = np.diag(np.nanvar(self.data, axis = 0))

    #     # missing_indices = np.argwhere(np.isnan(self.data))
    #     # Start updating
    #     Mu_tilde = {}
    #     S_tilde = np.zeros((num_rows, num_columns, num_columns))
    #     X_tilde = self.data.copy()

    #     # for j in range(iterations):
    #     #     S_tilde[:] = 0  # Reset S_tilde for each iteration
    #     #     M_i = rows_w_missing[rows_w_missing != -1]
    #     #     O_i = rows_w_observed[rows_w_observed != -1]

    #     #     # get submatrices of estimated covariance matrix
    #     #     S_MM = cov_matrix[np.ix_(M_i, M_i)]
    #     #     S_MO = cov_matrix[np.ix_(M_i, O_i)]
    #     #     S_OM = S_MO.T
    #     #     S_OO = cov_matrix[np.ix_(O_i, O_i)]

    #     #     # Calculate x tilde/contribution of x tilde to mu
    #     #     Mu_tilde['values'] = Mu[np.ix_(M_i)] + \
    #     #         S_MO @ np.linalg.inv(S_OO) @ \
    #     #         (X_tilde[:, O_i] - Mu[np.ix_(O_i)]).T
    #     #     X_tilde[:, M_i] = Mu_tilde['values'].T

    #     #     # Calculate contribution of x tilde to cov matrix
    #     #     S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
    #     #     S_tilde[:, np.ix_(M_i, M_i)] = S_MM_O

    #     #     Mu_new = np.mean(X_tilde, axis=0)
    #     #     S_new = np.cov(X_tilde.T, bias=True) + S_tilde.sum(0) / num_rows

    #     #     no_conv = \
    #     #         np.linalg.norm(Mu - Mu_new) >= stop or \
    #     #         np.linalg.norm(cov_matrix - S_new, ord=2) >= stop
    #     #     Mu = Mu_new
    #     #     cov_matrix = S_new
    #     #     np.linalg.norm(X_tilde-self.data)
    #     #     if not no_conv:
    #     #         print(f"Stopping after {j} iterations")
    #     #         break

    #     for j in range(iterations):
    #         for i in filter(lambda x: mask_missing[x].sum() < num_columns, range(num_rows)):
    #             S_tilde[i] = np.zeros_like(S_tilde[i])
    #             M_i, O_i = rows_w_missing[i, ][rows_w_missing[i, ] != -1], rows_w_observed[i, ][rows_w_observed[i, ] != -1]
    #             # get submatrices of estimated covariance matrix
    #             S_MM = cov_matrix[np.ix_(M_i, M_i)]
    #             S_MO = cov_matrix[np.ix_(M_i, O_i)]
    #             S_OM = S_MO.T
    #             S_OO = cov_matrix[np.ix_(O_i, O_i)]
    #             # Calculate x tilde/contribution of x tilde to mu
    #             Mu_tilde[i] = Mu[np.ix_(M_i)] +\
    #                 S_MO @ np.linalg.inv(S_OO) @\
    #                 (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
    #             X_tilde[i, M_i] = Mu_tilde[i]
    #             # Calculate contribution of x tilde to cov matrix
    #             S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
    #             S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O

    #         Mu_new = np.mean(X_tilde, axis = 0)
    #         S_new = np.cov(X_tilde.T, bias = True) +  S_tilde.sum(0) / num_rows


    #         no_conv =\
    #             np.linalg.norm(Mu - Mu_new) >= stop or\
    #             np.linalg.norm(cov_matrix - S_new, ord = 2) >= stop
    #         Mu = Mu_new
    #         cov_matrix = S_new

    #         if not no_conv:
    #             print(f"Stopping after {j} iterations")
    #             break


    #     self.mu = Mu
    #     self.cov = cov_matrix
    #     self.missing_cols = mask_missing
    #     print("\a completed training ")

    #     return X_tilde

    def fit_transform(self, X, eps = 1e-08):
        '''(np.array, int, number) -> {str: np.array or int}

        Precondition: max_iter >= 1 and eps > 0

        Return the dictionary with five keys where:
        - Key 'mu' stores the mean estimate of the imputed data.
        - Key 'Sigma' stores the variance estimate of the imputed data.
        - Key 'X_imputed' stores the imputed data that is mutated from X using
        the EM algorithm.
        - Key 'C' stores the np.array that specifies the original missing entries
        of X.
        - Key 'iteration' stores the number of iteration used to compute
        'X_imputed' based on max_iter and eps specified.
        '''
        max_iter = self.max_iter
        nr, nc = X.shape
        C = np.isnan(X) == False

        # Collect M_i and O_i's
        one_to_nc = np.arange(1, nc + 1, step = 1)
        M = one_to_nc * (C == False) - 1
        O = one_to_nc * C - 1

        # Generate Mu_0 and Sigma_0
        Mu = np.nanmean(X, axis = 0)
        observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
        S = np.cov(X[observed_rows, ].T)

        observed_rows = np.where(~np.isnan(sum(X.T)))[0]
        if observed_rows.size > 0:
            cov_matrix = np.cov(X[observed_rows, ].T)
        else:
            cov_matrix = np.diag(np.nanvar(X, axis = 0))

        if np.isnan(cov_matrix).any():
            cov_matrix = np.diag(np.nanvar(X, axis = 0))
        # Start updating
        Mu_tilde, S_tilde = {}, {}
        X_tilde = X.copy()
        no_conv = True
        iteration = 0
        while no_conv and iteration < max_iter:
            for i in range(nr):
                S_tilde[i] = np.zeros(nc ** 2).reshape(nc, nc)
                if set(O[i, ]) != set(one_to_nc - 1): # missing component exists
                    M_i, O_i = M[i, ][M[i, ] != -1], O[i, ][O[i, ] != -1]
                    S_MM = S[np.ix_(M_i, M_i)]
                    S_MO = S[np.ix_(M_i, O_i)]
                    S_OM = S_MO.T
                    S_OO = S[np.ix_(O_i, O_i)]
                    Mu_tilde[i] = Mu[np.ix_(M_i)] +\
                        S_MO @ np.linalg.inv(S_OO) @\
                        (X_tilde[i, O_i] - Mu[np.ix_(O_i)])
                    X_tilde[i, M_i] = Mu_tilde[i]
                    S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                    S_tilde[i][np.ix_(M_i, M_i)] = S_MM_O
            Mu_new = np.mean(X_tilde, axis = 0)
            S_new = np.cov(X_tilde.T, bias = 1) +\
                reduce(np.add, S_tilde.values()) / nr
            no_conv =\
                np.linalg.norm(Mu - Mu_new) >= eps or\
                np.linalg.norm(S - S_new, ord = 2) >= eps
            Mu = Mu_new
            S = S_new
            iteration += 1

        result = {
            'mu': Mu,
            'Sigma': S,
            'X_imputed': X_tilde,
            'C': C,
            'iteration': iteration
        }

        return X_tilde


def gaussian_linspace(mean: float, std: float):
    low = mean - 6 * std
    high = mean + 6 * std
    data = np.linspace(low, high, 1000)
    const = 1 / (std * np.sqrt(2 * np.pi))
    gauss = const * np.exp(-1 * np.square(data - mean) / (2 * std**2))
    return data, gauss




if __name__ == "__main__":
    # data, labels = gen_data(2, 4, -10, 10, 40, 200)
    # a = ExpectationMaximization().fit(data, labels).predict(4)
    # a.plot_2d()
    from utils import read_missing
    data = read_missing("./missing/MissingData2.txt").T
    # print(data.shape)
    imputed = EMImputer().fit_transform(data.values, iterations=1000)