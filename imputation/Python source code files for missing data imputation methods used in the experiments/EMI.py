#
from __future__ import division 
import scipy.stats as sc_stats
from itertools import permutations,combinations

from pytesmo.metrics import rmsd
from pytesmo.metrics import nrmsd
from pytesmo.metrics import aad
#

import numpy as np

import pandas as pd

import time

#from fancyimpute import IterativeImputer

#from ycimpute.imputer import EM

import impyute as impy

from openpyxl import load_workbook


#from ycimpute.imputer import KNN

complete_data=pd.read_csv('Original-DataSet-Without-Missing.csv')
df_complete_data=pd.DataFrame(data=complete_data, columns=complete_data.columns, index=complete_data.index)

numvar= len(df_complete_data.columns)# num of columns

X_incomplete=pd.read_csv('1% #0- Dataset-With-Missing.csv')
df_incomplete_data=pd.DataFrame(data=X_incomplete, columns=X_incomplete.columns, index=X_incomplete.index)


############################
start_time = time.time()

#X_filled_IterativeImputer = EM().complete(df_incomplete_data.values)
X_filled_IterativeImputer = impy.em(df_incomplete_data.values)
#X_filled_IterativeImputer = KNN(k=3).complete(df_incomplete_data.values)

#complete_data_, _, _ = min_max_scale(complete_data)
#X_filled, _, _ = min_max_scale(X_filled)


#X_filled_IterativeImputer = IterativeImputer().fit_transform(X_incomplete)
###
timeFinal = (time.time() - start_time)
print("--- %s seconds ---" % timeFinal)
##############################

df_after_IterativeImputer=pd.DataFrame(data=X_filled_IterativeImputer, columns=X_incomplete.columns, index=X_incomplete.index)

df_after_IterativeImputer.to_csv("1% #6- Dataset-Without-Missing-After-EMI.csv",index=False ,sep=';', encoding='utf-8')

# print root mean squared error for imputation method above
print("^^^^^^^^^^^^^^^^^^RMSE^^^^^^^^^^^^^^^^^^")
sum_of_RMSE = 0
sum_of_NRMSE = 0
sum_of_AE = 0
for i in range(0,numvar):
	#print(i)
	#print("----------------------")
	l1_after_imputation = df_after_IterativeImputer.iloc[:,i].values
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

ws1['AR6'] = average_RMSE_All_Cols
ws1['AS6'] = average_NRMSE_All_Cols
ws1['AT6'] = average_AE_All_Cols
ws1['AU6'] = timeFinal

wb.save(summaryName)



