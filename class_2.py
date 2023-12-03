import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_classification_dataset, features_histograms_mean_std
# import sklearn
from sklearnex import patch_sklearn
from sklearn.calibration import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC


import joblib

patch_sklearn()
model_3 = Pipeline([
      ('Imputer', KNNImputer(n_neighbors=3, weights='distance')),
      ('Scaler', StandardScaler()),
      ('PCA', PCA(n_components=18)),
      ('SVC', SVC(verbose=True, random_state=12))
    ])

param_grid_3 = [
    {
        # "Imputer": [KNNImputer(weights='distance')],
        # "Imputer__n_neighbors":[1,3,7,21],
       "SVC__C":[0.1, 10, 100, 1000],
       "SVC__kernel":["linear", "rbf"],
    #    "SVC__degree":[2,3],
       "SVC__decision_function_shape":["ovr"]


    },
]


train, target, test = read_classification_dataset(2)
X, y = train.values, target.values.flatten()
grid_3 = GridSearchCV(model_3, param_grid_3, scoring="f1_macro", cv=3)
grid_3.fit(X, y)
print(grid_3.best_score_)
print(grid_3.best_params_)
best_model_3 = grid_3.best_estimator_
joblib.dump(best_model_3, "c_2_KNNI(3)_Scaler_PCA(18)_SVC.pkl")


