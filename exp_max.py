import numpy as np

rng = np.random.default_rng()

def expectation(
    data:np.ndarray,
    means:np.ndarray,
    variances:np.ndarray)-> np.ndarray:
    # one row for each data point
    # row contains a value for each hidden z
    expect_z = np.ndarray((len(data), len(means)), np.float64)
    for i, (mean, variance) in enumerate(zip(means, variances)):
        expect_z[:,i] = np.exp((-1*(data-mean)**2)/(2*variance))

    # proportion of likelihood to total
    # expectation of z
    # len(data) x len(means)
    return (expect_z.T*(1/(expect_z.sum(1)+1e-230))).T

def new_hyp(
    data:np.ndarray,
    means:np.ndarray,
    variances:np.ndarray)->tuple[np.ndarray, ...]:

    expect = expectation(data,means,variances)
    # sum of weights for each hidden variable
    sum_weights:np.ndarray = expect.T.sum(axis=1)
    # weighted sum of data over sum of weights for each hidden variab;e
    new_means:np.ndarray = np.dot(expect.T, data) / sum_weights
    # element wise sqared difference between data and estimated means
    sq_diff:np.ndarray = np.square(data.reshape(len(data),1) - \
        new_means.reshape(1,len(variances)))
    # reweighting squared difference
    mse_weighted:np.ndarray = np.multiply(expect, sq_diff)
    # estimated variance for each distribution
    new_variance:np.ndarray = mse_weighted.sum(axis=0) / sum_weights

    return new_means, new_variance


def expectation_max(
    data:np.ndarray,
    number_of_distributions:int,
    stop:float=1e-7)->np.ndarray:

    means = rng.random(number_of_distributions, np.float64) + data.mean()
    variance = rng.random(number_of_distributions, np.float64) + data.var()
    means.sort()

    k = 2
    # stop when there is less than stop=1e-7 distance between new and old
    # in means and variance
    while k > stop:
        # get new mean/var
        new_means, new_variance = new_hyp(data, means, variance)
        # find change
        diff = new_means-means
        diff_v = new_variance-variance
        k = max((np.sqrt(np.dot(diff, diff))/2, np.sqrt(np.dot(diff_v, diff_v))/2))
        # update means/var
        means, variance = new_means, new_variance
    return np.array(sorted(zip(means, np.sqrt(variance))))


def gaussian_linspace(mean:float, std:float):
    low = mean - 6*std
    high = mean + 6*std
    data = np.linspace(low, high, 1000)
    const = 1/(std*np.sqrt(2*np.pi))
    gauss = const*np.exp(-np.square(data-mean)/(2*std**2))
    return data, gauss