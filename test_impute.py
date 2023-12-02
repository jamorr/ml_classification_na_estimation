import os

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

import exp_max
import soft_impute
import svd_impute
import utils


def rmse(d1, d2):
    return np.linalg.norm(d1 - d2, ord=2)

def nrmse(d1, d2):
    diff = d1 - d2
    msse = (diff.T@diff)/len(diff)
    vari = np.var(d2)
    return np.sqrt(msse/vari)

def create_MAR(
    data: np.ndarray,
    true_missing: np.ndarray,
    missing_rate: float = 0.03,
) -> np.ndarray:
    data_missing = data.copy()
    _, cols = data_missing.shape
    keep_mask = np.ones_like(data)
    remove_num = int(missing_rate * data.size)
    if remove_num < data.size - remove_num:
        indices = [
            i
            for (i, _), ma in zip(np.ndenumerate(data_missing), ~true_missing.ravel())
            if ma
        ]
    else:
        indices = [i for (i, _) in np.ndenumerate(data_missing)]

    # keep_mask = np.random.random(rows * cols).reshape(rows, cols) > missing_rate
    indices_mask = np.random.choice(
        range(len(indices)),
        size=remove_num,
        replace=False,
    )

    for i in indices_mask:
        keep_mask[indices[i]] = False
    keep_mask = keep_mask.astype(bool)

    # Check for which i's we have all components become missing
    checker = np.where(sum(keep_mask.T) == 0)[0]
    if len(checker) == 0:
        # Every X_i has at least one component that is observed,
        # which is what we want
        data_missing[~keep_mask] = np.nan
    else:
        # Otherwise, randomly "revive" some components in such X_i's
        for index in checker:
            reviving_components = np.random.choice(
                cols, int(np.ceil(cols * np.random.random())), replace=False
            )
            keep_mask[index, np.ix_(reviving_components)] = True
        data_missing[~keep_mask] = np.nan

    return data_missing


def iterative_impute_test(
    data: np.ndarray, imputer, n_iterations: int = 2, scoring=rmse, **kwargs
) -> list:
    # missing_indices = np.argwhere(np.isnan(data))
    missing_mask = np.isnan(data)
    last_impute = imputed_data = imputer(**kwargs).fit_transform(data.copy())
    percent_missing = missing_mask.sum().sum() / missing_mask.size
    print(f"{percent_missing*100:.3f}% missing...", end="")

    scores = []
    for _ in range(n_iterations):
        removed = create_MAR(last_impute.copy(), missing_mask, percent_missing)
        new_imputed = imputer(**kwargs).fit_transform(removed)
        score = scoring(new_imputed[~missing_mask], data[~missing_mask])
        print(score)
        scores.append(score)
        last_impute = new_imputed

    return scores, imputed_data


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    datasets = {}
    datasets[1] = utils.read_missing("./missing/MissingData1.txt").T
    datasets[2] = utils.read_missing("./missing/MissingData2.txt").T
    datasets[3] = utils.read_missing("./missing/MissingData3.txt").T


    impute_methods = {
        # IterativeImputer:{},
        soft_impute.SoftImputer: {"max_iterations":1000,"lmbd":0.07},
        svd_impute.SVDImputer: {"max_iterations":1000},
        KNNImputer: {"n_neighbors": 2, "weights": "distance"},
        SimpleImputer: {"strategy": "mean"},
        # exp_max.EMImputer:{"max_iter":10},
    }
    best_scores = [0.1025, 0.143, 0.66092886]
    for k, v in datasets.items():
        print(f"MissingData{k}:")
        eigen = np.sqrt(np.abs(np.linalg.eigvals(v.cov().values)))
        eigen = eigen/eigen.sum()
        entropy = -sum([e*np.log(e) for e in eigen[np.nonzero(eigen)]])/np.log(np.count_nonzero(eigen))
        print("Covariance Matrix Entropy: ", entropy)
        scores = {}
        best_score = best_scores[k-1]
        for imputer, kwargs in impute_methods.items():
            print(f"Starting {imputer.__name__}... ", end="")

            try:
                scores[imputer.__name__], candidate = iterative_impute_test(
                    v.values, imputer, n_iterations=2,scoring=nrmse, **kwargs
                )

            except np.linalg.LinAlgError:
                print("Failed :(")
                continue
            else:
                print("Completed!")
            if scores[imputer.__name__][0] < best_score:
                np.savetxt(f"./predictions/MissingData{k}.txt", candidate, delimiter="\t")

        print("RMSE")
        print(pd.DataFrame(scores))
    # classify_datasets = {}
    # classify_datasets[1] = utils.read_classification_dataset(1)[0].values
    # classify_datasets[2] = utils.read_classification_dataset(2)[0].values
    # for k, v in classify_datasets.items():
    #     print(f"ClassifyData{k} Train:")
    #     scores = {}
    #     for imputer, kwargs in impute_methods.items():
    #         print(f"Starting {imputer.__name__}... ", end="")
    #         try:
    #             scores[imputer.__name__], candidate = iterative_impute_test(
    #                 v, imputer, n_iterations=2, scoring=nrmse,**kwargs
    #             )
    #         except np.linalg.LinAlgError:
    #             print("Failed :(")
    #             continue
    #         else:
    #             print("Completed!")
    #     print("RMSE")
    #     print(pd.DataFrame(scores))


if __name__ == "__main__":
    main()
