import joblib
import exp_max
import soft_impute
from sklearn.impute import KNNImputer, SimpleImputer
import pandas as pd
import numpy as np
import utils
import os



def rmse(d1, d2):
    return np.linalg.norm(d1-d2,ord='fro')

def create_MAR(data:np.ndarray, true_missing:np.ndarray, missing_rate:float=0.03,)->np.ndarray:
    data_missing = data.copy()
    _, cols = data_missing.shape
    keep_mask = np.ones_like(data)
    remove_num = int(missing_rate*data.size)
    if remove_num < data.size - remove_num:
        indices = [i for (i,_),ma in zip(np.ndenumerate(data_missing), ~true_missing.ravel()) if ma]
    else:
        indices = [i for (i,_) in np.ndenumerate(data_missing)]

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
                cols,
                int(np.ceil(cols * np.random.random())),
                replace = False
            )
            keep_mask[index, np.ix_(reviving_components)] = True
        data_missing[~keep_mask] = np.nan

    return data_missing


def iterative_impute_test(data:np.ndarray, imputer, n_iterations:int=2, **kwargs)->list:
    # missing_indices = np.argwhere(np.isnan(data))
    missing_mask = np.isnan(data)
    last_impute = imputed_data = imputer(**kwargs).fit_transform(data.copy())
    scores = []
    for _ in range(n_iterations):
        new_imputed = imputer(**kwargs).fit_transform(create_MAR(last_impute.copy(), missing_mask, missing_mask.sum().sum()/missing_mask.size))
        scores.append(rmse(new_imputed, imputed_data))
        last_impute = new_imputed

    return scores

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    datasets = {}
    datasets[1] = utils.read_missing("./missing/MissingData1.txt").T.values
    datasets[2] = utils.read_missing("./missing/MissingData2.txt").T.values
    datasets[3] = utils.read_missing("./missing/MissingData3.txt").T.values
    impute_methods = {
        SimpleImputer:{"strategy":"mean"},
        KNNImputer:{"n_neighbors":1,'weights':'distance'},
        soft_impute.SoftImputer:{},
        # exp_max.EMImputer:{"max_iter":10},
    }
    for k, v in datasets.items():
        print(f"MissingData{k}:")
        scores = {}
        for imputer, kwargs in impute_methods.items():
            try:
                scores[imputer.__name__] = iterative_impute_test(v, imputer, n_iterations=2, **kwargs)
            except np.linalg.LinAlgError:
                continue
        print("RMSE")
        print(pd.DataFrame(scores))

if __name__ == "__main__":
    main()