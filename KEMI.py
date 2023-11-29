#
from __future__ import division

from pytesmo.metrics import rmsd
from pytesmo.metrics import nrmsd
from pytesmo.metrics import aad
#
from math import sqrt

import numpy as np

import random

#from ycimpute.imputer import EM
import pandas as pd

import time

# import impyute as impy
from exp_max import EMImputer

from fancyimpute import SoftImpute
from fancyimpute import IterativeImputer

from openpyxl import load_workbook

import utils

##############################################

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = np.sqrt(np.nansum((np.array(test_row)-np.array(train_row))**2))
        #dist = dist1.euclidean(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


##############################################
def kEMI(complete_data , X_incomplete):
#complete_data=pd.read_csv('Sonar_Without_Missing.csv')

#X_incomplete=pd.read_csv('Sonar_With_Missing_MCAR.csv')

#df_complete_data=pd.DataFrame(data=complete_data, columns=complete_data.columns, index=complete_data.index)

    #numvar= len(df_complete_data.columns)# num of columns

    df_incomplete_data=pd.DataFrame(data=X_incomplete, columns=X_incomplete.columns, index=X_incomplete.index)

    #Split X into Xobs and Xmis subsets

    X_missing = df_incomplete_data[df_incomplete_data.isnull().any(axis=1)] #Xmis is the incomplete subset of X

    #print(Xmis.shape)

    X_observed = df_incomplete_data.dropna(inplace=False) #Xobs is the complete subset of X

    #print(Xobs.shape)

    Pt = X_observed #Pt is the pool that is initialized to Xobs

    while not(X_missing.empty):
        #print("------------------------------------------------------------------------")
        # Select an incomplete record from Xmis and add it into the pool
        xi = X_missing.iloc[0]

        #get index of xi to put the imputed xi in correct order in dataset
        index_of_xi = X_missing.index.tolist()[0]

        #print(xi)
        Pt = pd.concat([Pt,xi])

        # Create a complete subset by excluding incomplete features with missing value
        col_index_removed=[]
        for j in range(len(Pt.columns)):
            if pd.isnull(xi[j]):
                col_index_removed.append(j)

        #print(col_index_removed)

        St = Pt.drop(Pt.columns[col_index_removed], axis=1, inplace=False)

        St_Complete_Temp = St.copy()

        # Induce a random missing into xiz
        #St.iloc[len(St.index)-1 , 20]=np.NaN
        random_missing_col = random.randint(0,len(St.columns)-1)
        St.iloc[len(St.index)-1 , random_missing_col]=np.NaN


        mp = len(Pt.index) # mp is the number of records in Pt

        K_List = []
        RMSE_List = []
        for k_count in range(2,mp):
            K_List.append(k_count)
            ###knni###
            neighbors_for_xi_in_St = get_neighbors(St.iloc[0:len(St.index)-1].values.tolist(), St.iloc[len(St.index)-1].values.tolist(), k_count)
            df_neighbors_for_xi_in_St = pd.DataFrame(data=neighbors_for_xi_in_St, columns=St.columns)

            one_col_df_neighbors_for_xi_in_St = df_neighbors_for_xi_in_St.iloc[:, [random_missing_col]]

            imputed_value = one_col_df_neighbors_for_xi_in_St.mean()#[0]

            actual_value = St_Complete_Temp.iloc[len(St.index)-1 , random_missing_col]

            average_RMSE_All_Cols_Final = np.sqrt(np.mean((imputed_value-actual_value)**2))

            RMSE_List.append(average_RMSE_All_Cols_Final)

            #break;

            ######

        #print(K_List)
        #print(RMSE_List)
        index_of_min_value_in_RMSE_List = RMSE_List.index(min(RMSE_List))

        k_of_min_RMSE = K_List[index_of_min_value_in_RMSE_List]
        #min_RMSE = RMSE_List[index_of_min_value_in_RMSE_List]

        #print(k_of_min_RMSE)
        #print(min_RMSE)

        neighbors_αi = get_neighbors(Pt.iloc[0:len(Pt.index)-1].values.tolist(), xi.values.tolist(), k_of_min_RMSE)
        df_neighbors_αi = pd.DataFrame(data=neighbors_αi, columns=Pt.columns)

        Θi = pd.concat([df_neighbors_αi,xi])

        ####EMI####
        Θi_filled_EM = EMImputer().fit_transform(Θi.values, iterations=50)
        #Θi_filled_EM = SoftImpute().fit_transform(Θi.values)
        #Θi_filled_EM = IterativeImputer().fit_transform(Θi.values)
        ###########

        df_Θi_filled_EM=pd.DataFrame(data=Θi_filled_EM, columns=Pt.columns)

        xi_imputed = df_Θi_filled_EM.iloc[len(df_Θi_filled_EM.index)-1]

        xi_imputed_with_index = pd.Series(xi_imputed).rename(index_of_xi)

        X_observed = pd.concat([X_observed, xi_imputed_with_index])

        X_missing = X_missing.iloc[1:]

        Pt = X_observed

        break

    X_observed.sort_index(inplace=True)
    all_dataset_imputed = X_observed

    return all_dataset_imputed

##############################################

# complete_data=pd.read_csv('Original-DataSet-Without-Missing.csv')
# df_complete_data=pd.DataFrame(data=complete_data, columns=complete_data.columns, index=complete_data.index)

# numvar= len(df_complete_data.columns)# num of columns


# X_incomplete=pd.read_csv('../missing/MissingData')
X_incomplete = utils.read_missing("./missing/MissingData2.txt").T
df_incomplete_data=pd.DataFrame(data=X_incomplete, columns=X_incomplete.columns, index=X_incomplete.index)
numvar= len(X_incomplete.columns)# num of columns

start_time = time.time()

all_dataset_imputed = kEMI(X_incomplete , X_incomplete)

timeFinal = (time.time() - start_time)
print("--- %s seconds ---" % timeFinal)

all_dataset_imputed.to_csv("./predictions/MissingData2_Imputed.csv",index=False ,sep='\t', encoding='utf-8')

# print("^^^^^^^^^^^^^^^^^^RMSE^^^^^^^^^^^^^^^^^^")
# sum_of_RMSE = 0
# sum_of_NRMSE = 0
# sum_of_AE = 0
# for i in range(0,numvar):
# 	#print(i)
# 	#print("----------------------")
# 	l1_after_imputation = all_dataset_imputed.iloc[:,i].values
# 	#print(l1_after_imputation)
# 	l2_complete_data = df_complete_data.iloc[:,i].values
# 	#print("**********************")
# 	#print(l2_complete_data)
# 	#print("%%%%%%%%%%%%%%%%%%%%%%")
	rms = sqrt(mean_squared_error(l2_complete_data, l1_after_imputation))
# 	rms = rmsd(l2_complete_data, l1_after_imputation) #Root-mean-square deviation (RMSD)
# 	nrms = nrmsd(l2_complete_data, l1_after_imputation) #Normalized root-mean-square deviation (nRMSD)
# 	ae = aad(l2_complete_data, l1_after_imputation) #Average (=mean) absolute deviation (AAD).

# 	sum_of_RMSE = sum_of_RMSE + rms
# 	sum_of_NRMSE = sum_of_NRMSE + nrms
# 	sum_of_AE = sum_of_AE + ae

# 	print("RMSE Col",i,":",rms)
# 	print("NRMSE Col",i,":",nrms)
# 	print("AE Col",i,":",ae)
# 	print("----------------------")


# average_RMSE_All_Cols = sum_of_RMSE / numvar
# average_NRMSE_All_Cols = sum_of_NRMSE / numvar
# average_AE_All_Cols = sum_of_AE / numvar
# print("RMSE For All Cols: ",average_RMSE_All_Cols)
# print("NRMSE For All Cols: ",average_NRMSE_All_Cols)
# print("AE For All Cols: ",average_AE_All_Cols)
# print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


# summaryName = "Summary.xlsx"
# wb = load_workbook(summaryName)
# ws1 = wb["Sheet1"]

# ws1['BD6'] = average_RMSE_All_Cols
# ws1['BE6'] = average_NRMSE_All_Cols
# ws1['BF6'] = average_AE_All_Cols
# ws1['BG6'] = timeFinal

# wb.save(summaryName)

