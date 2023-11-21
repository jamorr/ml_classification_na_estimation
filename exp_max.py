from itertools import chain
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal  # for generating pdf
from sklearn.datasets import make_spd_matrix

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

def gaussian_linspace(mean: float, std: float):
    low = mean - 6 * std
    high = mean + 6 * std
    data = np.linspace(low, high, 1000)
    const = 1 / (std * np.sqrt(2 * np.pi))
    gauss = const * np.exp(-1 * np.square(data - mean) / (2 * std**2))
    return data, gauss




if __name__ == "__main__":
    data, labels = gen_data(2, 4, -10, 10, 40, 200)
    a = ExpectationMaximization().fit(data, labels).predict(4)
    a.plot_2d()