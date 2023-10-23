import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal  # for generating pdf

rng = np.random.default_rng()

def plot_cluster_2d(data:np.ndarray,  means:np.ndarray, covar:np.ndarray, labels: np.ndarray | None = None):
    X1 = np.linspace(data[:,0].min()-1, data[:,0].max()+1, 250)
    X2 = np.linspace(data[:,1].min()-1, data[:,1].max()+1, 250)
    X, Y = np.meshgrid(X1, X2)
    distributions = []
    for m, c in zip(means, covar):
        distributions.append(multivariate_normal(m, c))

    pos = np.empty(X.shape + (2,))                # a new array of given shape and type, without initializing entries
    pos[:, :, 0],pos[:, :, 1] = X, Y

    plt.figure(figsize=(10,10))
    # creating the figure and assigning the size
    if labels:
        print("Not implemented")
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
    data: np.ndarray, means: np.ndarray, variances: np.ndarray, pi_hat: np.ndarray
) -> tuple[np.ndarray, ...]:
    n_dist =  len(means)
    expect_z = np.ndarray((len(data),n_dist), np.float64)
    for i, (mean, variance) in enumerate(zip(means, variances)):
        expect_z[:, i] = pi_hat[i] * multivariate_normal.pdf(data, mean, variance, allow_singular=True)
    likelihood = np.sum(np.log(expect_z.sum(0)))
    # pi hat is the predicted proportion of points which are part of each hidden variable
    # sum weights of each hidden variable as a proportion of the
    # total number of weights for all hidden variables (of the same category like clusters)
    return (expect_z.T * (1 / (expect_z.sum(1) + 1e-230))).T, likelihood


def new_hyp_mutli_var(
    data: np.ndarray, means: np.ndarray, variances: np.ndarray, pi_hat: np.ndarray
) -> tuple[np.ndarray, ...]:
    # expectation values shape == len(data) x n_dist
    expect, likelihood = expectation_multi_var(data, means, variances, pi_hat)

    # sum of weights for each hidden variable
    # weighted sum of data over sum of weights for each hidden variab;e
    new_means: np.ndarray = np.array(
        [
            np.average(
                data, axis=0, weights=e
            )
            for e in expect.T
        ]
    )
    new_cov:np.ndarray = np.array(
        [
            np.cov(data.T, ddof = 0, aweights=e) for e in expect.T
        ]
    )
    new_pi_hat: np.ndarray = expect.sum(0) / len(data)
    return new_means, new_cov, new_pi_hat, likelihood


def expectation_max_multi_var(
    data: np.ndarray, number_of_distributions: int, stop: float = float("-inf"), iterations:int = 1000
) -> tuple[np.ndarray, ...]:
    # generate intial guesses for means and variances
    means = (
        rng.random((number_of_distributions, data.shape[1]), np.float64) + data.mean()
    )
    variance = rng.random(
        (number_of_distributions, data.shape[1]), np.float64
    )
    variance = np.array([np.outer(v,v) for v in variance])
    for _ in range(number_of_distributions-1):
        vv = rng.random(
        (number_of_distributions, data.shape[1]), np.float64)
        variance += np.array([np.outer(v,v) for v in vv])

    pi_hat = (
        np.ones(number_of_distributions) * (1 / number_of_distributions) * len(data)
    )

    means.sort()
    last = 0
    k = 2
    # stop when there is less than stop=1e-7 distance between new and old
    # in means and variance
    i = 0
    while k > stop and i < iterations:
        # get new mean/var
        means, variance, pi_hat, likelihood = new_hyp_mutli_var(data, means, variance, pi_hat)
        k, last = likelihood - last, likelihood
        i += 1
    return means, variance


def gaussian_linspace(mean: float, std: float):
    low = mean - 6 * std
    high = mean + 6 * std
    data = np.linspace(low, high, 1000)
    const = 1 / (std * np.sqrt(2 * np.pi))
    gauss = const * np.exp(-1 * np.square(data - mean) / (2 * std**2))
    return data, gauss
