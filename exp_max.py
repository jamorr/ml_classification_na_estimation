from itertools import chain
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal  # for generating pdf

rng = np.random.default_rng()

def plot_cluster_2d(data:np.ndarray,  means:np.ndarray, covar:np.ndarray, labels: np.ndarray | None = None):
    X1 = np.linspace(data[:,0].min()-1, data[:,0].max()+1, 250)
    X2 = np.linspace(data[:,1].min()-1, data[:,1].max()+1, 250)
    X, Y = np.meshgrid(X1, X2)
    distributions = []
    for m, c in zip(means, covar):
        distributions.append(multivariate_normal(m, c, allow_singular=True))

    pos = np.empty(X.shape + (2,))                # a new array of given shape and type, without initializing entries
    pos[:, :, 0], pos[:, :, 1] = X, Y

    # plt.figure(figsize=(10,10))
    # creating the figure and assigning the size
    if isinstance(labels, np.ndarray) and labels.any():
        groups = pd.DataFrame(np.concatenate((data,labels), axis=1))
        groups.plot.scatter(x=0, y=1, c=2)
    else:
        plt.scatter(data[:,0], data[:,1], marker='o') # type: ignore

    colors = ['r','g','b','k','c', 'm']
    for dist, col in zip(distributions, colors):
        plt.contour(X, Y, dist.pdf(pos), colors=col ,alpha = 0.5)
    plt.axis('equal')                                                                  # making both the axis equal
    plt.xlabel('X-Axis', fontsize=16)                                                  # X-Axis
    plt.ylabel('Y-Axis', fontsize=16)                                                  # Y-Axis
    plt.grid()                                                                         # displaying gridlines
    plt.show()

def expectation_multi_var(
    data: np.ndarray, means: np.ndarray, variances: np.ndarray, proportions: np.ndarray, estimate_proportion:bool
) -> tuple[np.ndarray, float]:
    n_dist =  len(means)
    expect_z = np.ndarray((len(data),n_dist), np.float64)


    if estimate_proportion:
        for i, (mean, variance) in enumerate(zip(means, variances)):
            expect_z[:, i] = proportions[i] * multivariate_normal.pdf(data, mean, variance, allow_singular=True)
    else:
        for i, (mean, variance) in enumerate(zip(means, variances)):
            expect_z[:, i] = multivariate_normal.pdf(data, mean, variance, allow_singular=True)
    # likelihood = expect_z
    print(np.sum(np.log(expect_z+ 1e-230)).sum())
    expect_z = ((expect_z.T + 1e-230)* (1 / (expect_z.sum(1) + len(expect_z) * 1e-230))).T

    likelihood=12.5
    # pi hat is the predicted proportion of points which are part of each hidden variable
    # sum weights of each hidden variable as a proportion of the
    # total number of weights for all hidden variables (of the same category like clusters)
    return expect_z, likelihood


def new_hyp_mutli_var(
    data: np.ndarray, means: np.ndarray, variances: np.ndarray, proportions: np.ndarray, estimate_proportion:bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    # expectation values shape == len(data) x n_dist
    expect, likelihood = expectation_multi_var(data, means, variances, proportions, estimate_proportion)

    # sum of weights for each hidden variable
    # weighted sum of data over sum of weights for each hidden variable
    try:
        new_means: np.ndarray = np.array(
            [
                np.average(
                    data, axis=0, weights=e
                )
                for e in expect.T
            ]
        )
    except ZeroDivisionError as e:
        print(expect)
        raise e
    new_cov:np.ndarray = np.array(
        [
            np.cov(data.T, ddof = 0, aweights=e) for e in expect.T
        ]
    )
    new_pi_hat: np.ndarray = expect.sum(0) / len(data)
    return new_means, new_cov, new_pi_hat, expect, likelihood


def expectation_max_multi_var(
    data: np.ndarray, number_of_distributions: int, stop: float = float("-inf"), iterations:int = 1000, estimate_proportion:bool=True) -> tuple[np.ndarray, ...]:
    # generate intial guesses for means and variances
    means = (
        rng.random((number_of_distributions, data.shape[1]), np.float64) + data.mean()
    )
    # means = data[:number_of_distributions, :]
    variance = rng.random(
        (number_of_distributions, data.shape[1]), np.float64
    )
    variance = np.array([np.outer(v,v) for v in variance])
    for _ in range(number_of_distributions-1):
        vv = rng.random(
        (number_of_distributions, data.shape[1]), np.float64)
        variance += np.array([np.outer(v,v) for v in vv])

    proportions = (
        np.ones(number_of_distributions) * (1 / number_of_distributions) * len(data)
    )

    means.sort()
    last = 0
    k = 2
    # stop when there is less than stop=1e-7 distance between new and old
    # in means and variance
    i = 0
    expect = np.ndarray(1)
    while k > stop and i < iterations:
        # get new mean/var
        means, variance, proportions, expect, likelihood = new_hyp_mutli_var(data, means, variance, proportions, estimate_proportion)
        k, last = likelihood - last, likelihood
        i += 1

    return means, variance, expect

def em_imputer():
    return


def gaussian_linspace(mean: float, std: float):
    low = mean - 6 * std
    high = mean + 6 * std
    data = np.linspace(low, high, 1000)
    const = 1 / (std * np.sqrt(2 * np.pi))
    gauss = const * np.exp(-1 * np.square(data - mean) / (2 * std**2))
    return data, gauss

def main():
    from sklearn.datasets import make_spd_matrix

    num_clusters = 6
    num_dims = 2

    covars = np.array([make_spd_matrix(num_dims) for _ in range(num_clusters)])
    means = rng.uniform(low=-10, high=10, size=(num_clusters, num_dims))
    # print(covars.shape)
    # print(means.shape)
    clusters = []
    for m, cov in zip(means, covars):
        clusters.append(np.random.multivariate_normal(m, cov, size=(rng.integers(20,1000),)))
    gt_labels = np.array(list(chain.from_iterable([[[i]]*len(clusters[i]) for i in range(num_clusters)])))
    d = np.concatenate([c for c in clusters], axis=0)
    print(len(gt_labels))
    means, variance, labels = expectation_max_multi_var(d, 6, iterations=1000)
    plot_cluster_2d(d, means, variance, gt_labels)

if __name__ == "__main__":
    main()