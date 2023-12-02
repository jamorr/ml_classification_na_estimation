import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import KFold, cross_val_score,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
# from sklearn.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from utils import read_classification_dataset


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ConditionalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, imputer):
        self.imputer = imputer

    def fit(self, X, y=None):
        if np.isnan(X).any():
            self.imputer.fit(X, y)
        return self

    def transform(self, X, y=None):
        if np.isnan(X).any():
            return self.imputer.transform(X)
        return X
    
    
def modeltest(idnum:int):
    print(f"Finding best model for Classifier{idnum}")
    train,target,test = read_classification_dataset(idnum)
    imputed = SimpleImputer().fit_transform(train)

    X, y = imputed, target.values.flatten()
    candidates = [
        RandomForestClassifier(
          # max_depth=4,
          max_features="log2",
          bootstrap=True,
          # max_samples=1000,
          n_jobs=-1,
          class_weight="balanced_subsample",
          # warm_start=True,
          random_state=51,
          ),
        KNeighborsClassifier(weights="distance",),
        DecisionTreeClassifier(),
        SVC(),
    ]
    classifier_names = ["rforest","knn","dtree","SVC"]
    params = [
        #Random Forest
        {   "clf__max_depth":[None,1,2,3,4,5],
            # "Imputer":[SimpleImputer(),KNNImputer(weights="distance")],
        },
        #KNN
        {
        "clf__n_neighbors":[3,5,7,9,11],
        # "Imputer":[SimpleImputer(),KNNImputer(weights="distance")],

        },
        #Decision Tree
        {
        # "Imputer":[SimpleImputer(),KNNImputer(weights="distance")],
        "clf__max_depth":[None,1,2,3,4,5],
        "clf__splitter":["best","random"]
        },
        #Support Vector Classifier
        {
        # "Imputer":[SimpleImputer(),KNNImputer(weights="distance")],
        'clf__kernel':['linear','sigmoid','rbf','poly'],
        "clf__gamma":['scale','auto'],
        }
    ]
    best_model_score = -1  # Initialize with a score that will be definitely lower than any F1 score
    best_model_name = ""
    best_model_params = None

    for clf, par, name in zip(candidates, params, classifier_names):
        pipeline = Pipeline([
            ('Imputer', KNNImputer(weights="distance")), 
            ('Scaler', StandardScaler()), 
            ('clf', clf)
        ])
        grid = GridSearchCV(pipeline, param_grid=par, scoring='f1_weighted', cv=5)
        grid.fit(X, y=y)

        if grid.best_score_ > best_model_score:
            best_model_score = grid.best_score_
            best_model_name = name
            best_model_params = grid.best_params_
            best_model = grid.best_estimator_

    best_model_path = f'./models/c_{idnum}_best_model.pkl'
    joblib.dump(best_model, best_model_path)

    print(f"Best Model for Classification {idnum}: {best_model_name} with score {best_model_score}")
    print(f"Best Model Parameters: {best_model_params}\n")
    best_model.fit(train,target)
    np.savetxt(f"./predictions/TestLabel{idnum}.txt",best_model.predict(test),delimiter="\n")
    return best_model
if __name__ == "__main__":
    for i in [1,2,3,4,5]:
        modeltest(i)




