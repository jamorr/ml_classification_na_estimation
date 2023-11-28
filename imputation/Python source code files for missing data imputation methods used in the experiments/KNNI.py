#
from __future__ import division 
import scipy.stats as sc_stats
from itertools import permutations,combinations

from pytesmo.metrics import rmsd
from pytesmo.metrics import nrmsd
from pytesmo.metrics import aad

from sklearn.metrics import mean_absolute_error

#

import numpy as np

import pandas as pd

import time

from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

from sklearn import svm
##
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
##
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from openpyxl import load_workbook


##

##

##
#from fancyimpute import KNN
from sklearn.impute import KNNImputer
##

complete_data=pd.read_csv('Original-DataSet-Without-Missing.csv')
df_complete_data=pd.DataFrame(data=complete_data, columns=complete_data.columns, index=complete_data.index)

numvar= len(df_complete_data.columns)# num of columns

X_incomplete=pd.read_csv('1% #0- Dataset-With-Missing.csv')
############################
start_time = time.time()
###
#X_filled_knn = KNN(k=3).fit_transform(X_incomplete)

imputer = KNNImputer(n_neighbors=3, weights="uniform")
X_filled_knn = imputer.fit_transform(X_incomplete)
###
timeFinal = (time.time() - start_time)
print("--- %s seconds ---" % timeFinal)
##############################
df_after_knn=pd.DataFrame(data=X_filled_knn, columns=X_incomplete.columns, index=X_incomplete.index)

df_after_knn.to_csv("1% #2- Dataset-Without-Missing-After-KNN.csv",index=False ,sep=';', encoding='utf-8')

# print root mean squared error for imputation method above
print("^^^^^^^^^^^^^^^^^^RMSE^^^^^^^^^^^^^^^^^^")
sum_of_RMSE = 0
sum_of_NRMSE = 0
sum_of_AE = 0
for i in range(0,numvar):
	#print(i)
	#print("----------------------")
	l1_after_imputation = df_after_knn.iloc[:,i].values
	#print(l1_after_imputation)
	l2_complete_data = df_complete_data.iloc[:,i].values
	#print("**********************")
	#print(l2_complete_data)
	#print("%%%%%%%%%%%%%%%%%%%%%%")
	#rms = sqrt(mean_squared_error(l2_complete_data, l1_after_imputation))
	rms = rmsd(l2_complete_data, l1_after_imputation) #Root-mean-square deviation (RMSD)
	nrms = nrmsd(l2_complete_data, l1_after_imputation) #Normalized root-mean-square deviation (nRMSD)
	ae = aad(l2_complete_data, l1_after_imputation) #Average (=mean) absolute deviation (AAD).

	sum_of_RMSE = sum_of_RMSE + rms
	sum_of_NRMSE = sum_of_NRMSE + nrms
	sum_of_AE = sum_of_AE + ae

	print("RMSE Col",i,":",rms)
	print("NRMSE Col",i,":",nrms)
	print("AE Col",i,":",ae)
	print("----------------------")


average_RMSE_All_Cols = sum_of_RMSE / numvar
average_NRMSE_All_Cols = sum_of_NRMSE / numvar
average_AE_All_Cols = sum_of_AE / numvar
print("RMSE For All Cols: ",average_RMSE_All_Cols)
print("NRMSE For All Cols: ",average_NRMSE_All_Cols)
print("AE For All Cols: ",average_AE_All_Cols)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


summaryName = "Summary.xlsx"
wb = load_workbook(summaryName)
ws1 = wb["Sheet1"]

ws1['P6'] = average_RMSE_All_Cols
ws1['Q6'] = average_NRMSE_All_Cols
ws1['R6'] = average_AE_All_Cols
ws1['S6'] = timeFinal

wb.save(summaryName)


