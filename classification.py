import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_classification_dataset, features_histograms_mean_std
# import sklearn
from sklearnex import patch_sklearn

from sklearn.impute import KNNImputer
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering
from collections import Counter
# patch_sklearn()
import math
import numpy as np

def apply_mask(matrix, vector):
    # Ensure the input is a NumPy array
    matrix = np.array(matrix)
    vector = np.array(vector)

    # Check if the dimensions are compatible
    if matrix.shape[0] != vector.shape[0]:
        raise ValueError("Number of rows in matrix must be equal to the length of the vector")

    # Initialize an empty result vector
    result = []

    # Iterate through each column of the matrix
    for col_idx in range(matrix.shape[1]):
        result.append(np.ravel(vector[np.where(matrix[:, col_idx])]))

    return result

def entropy(*args)->float:
    # print(args)

    tot = sum(args)
    if not all(args):
        return 0
    out = 0
    for k in args:
        out += -1*(k/tot)*(math.log(k/tot, 2))
    return out


train, target, test = read_classification_dataset(1)
print(train)
X = train.values

imputer = KNNImputer(n_neighbors=3, weights='distance')
data = imputer.fit_transform(X)
print("impute done")
# print(target)
unique_labels =len(np.unique(target))
print("number of unique labels", unique_labels)
for num_clusters in range(unique_labels, unique_labels+20):
    model = SpectralBiclustering(n_clusters=num_clusters, random_state=0)
    model.fit(data)


    # print("clustering done")
    # print("bicluster shapes",[b.shape for b in model.biclusters_])
    # members = [[c.sum().sum() for c in b[::unique_labels]] for b in model.biclusters_]
    # print("bicluster members",members, sum(members[0]))
    masked_target = apply_mask(model.biclusters_[0][::num_clusters].T, target.values)
    # print(*masked_target, sep="\n")
    print(f"total entropy with {num_clusters} clusters:",
          sum([entropy(*Counter(cluster).values())*(len(cluster)/data.shape[0]) for cluster in masked_target]))

assert False
