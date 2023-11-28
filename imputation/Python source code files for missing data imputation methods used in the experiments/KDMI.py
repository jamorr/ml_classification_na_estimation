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

import impyute as impy

import math

from collections import defaultdict

#from fancyimpute import SoftImpute
#from fancyimpute import IterativeImputer

from chefboost import Chefboost as chef

from openpyxl import load_workbook



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

def reverse(data_list):
    return data_list[::-1]

    
##############################################
def KDMI(complete_data , X_incomplete):
    #complete_data=pd.read_csv('Sonar_Without_Missing.csv')
    
    #X_incomplete=pd.read_csv('Sonar_With_Missing_MCAR_NEW1.csv')
        
    #df_complete_data=pd.DataFrame(data=complete_data, columns=complete_data.columns, index=complete_data.index)
        
    #numvar= len(df_complete_data.columns)# num of columns
        
    df_incomplete_data=pd.DataFrame(data=X_incomplete, columns=X_incomplete.columns, index=X_incomplete.index)
    
    ######################################Step-1##############################################
    
    #Split DF into DI and DC subsets
    
    DI = df_incomplete_data[df_incomplete_data.isnull().any(axis=1)] #Xmis is the incomplete subset of X
    index_list_DI = DI.index
    
    DC = df_incomplete_data.dropna(inplace=False) #Xobs is the complete subset of X
    
    
    L = 0
    
    
    ######################################Step-2##############################################
    
    #total no. of attributes in DI having missing values
    M = DI.columns.get_indexer(DI.columns[DI.isnull().any()].tolist())
    
    #M = DI.columns[DI.isnull().any()].tolist()
    
    print(M)
    
    all_leaves = []
    L = 0
    Ij = {}
    
    for Ai in range(len(M)):
        DC_CPY = DC.copy()
        index_list = DC_CPY.index
        #print(index_list)
        #print(DC_CPY)
        min_value = min(DC_CPY[DC_CPY.columns[M[Ai]]])
        max_value = max(DC_CPY[DC_CPY.columns[M[Ai]]])
        domain_size = max_value - min_value;
        NC = math.sqrt(domain_size)
        #print(min_value)
        #print(max_value)
        #print(NC)
        max_range = max_value
        min_range = max_value * NC
        list_ranges = []
        list_ranges.append([min_range,max_range])
        while min_range >= min_value:
            max_range = min_range    
            min_range = max_range * NC 
            list_ranges.append([min_range,max_range])
        list_ranges = reverse(list_ranges)
        #print (list_ranges)
        DC_CPY['Decision'] = ['' for _ in range(len(DC_CPY))]
        for row in range(len(DC_CPY)):      
            for rg in list_ranges:
                if  rg[0] <= DC_CPY.iloc[row, M[Ai]] <= rg[1]:
                    DC_CPY.iloc[row, len(DC_CPY.columns)-1] = str(rg)
                    break
                
        DC_CPY.drop(DC_CPY.columns[M[Ai]], axis=1, inplace=True)
           
        #print(DC_CPY)
    
        #DC_CPY.rename(columns={'60':'Decision'}, inplace=True)
        #print(DC_CPY.iloc[row,:-1].tolist())
        
        DC_CPY_Temp = DC_CPY.copy()
        
        
        
        config = {'algorithm': 'C4.5'} #ID3, C4.5, CART, CHAID or Regression
        model = chef.fit(DC_CPY, config)
      
        #moduleName = "outputs/rules/rules" #this will load outputs/rules/rules.py
        #tree = chef.restoreTree(moduleName)
        
        #prediction = tree.findDecision(['Sunny', 'Hot', 'High', 'Weak'])
        chef.save_model(model, "model.pkl")
        #model = chef.load_model("model.pkl")
        
        #time.sleep(5) 
        
        
        
        modelnew = chef.load_model("model.pkl")
        leaves_one_missing_col = defaultdict(list)
          
        
        for row in range(len(DC_CPY_Temp)):
            prediction = chef.predict(modelnew,DC_CPY_Temp.iloc[row,:-1].tolist())
            leaves_one_missing_col[prediction].append(index_list[row])
            Ij[prediction] = False
            
    ######################################Step-3##############################################
    
        DI_temp = DI.drop(DI.columns[M[Ai]], axis=1, inplace=False)
        
        for row in range(len(DI)):
            if pd.isnull(DI.iloc[row, M[Ai]]):
                prediction = chef.predict(modelnew,DI_temp.iloc[row].tolist())
                leaves_one_missing_col[prediction].append(index_list_DI[row])
                #print(prediction)
                
        si = len(leaves_one_missing_col.keys())
        
        L = L + si
                
        all_leaves.append(leaves_one_missing_col)
        #print(all_leaves) 
        
        #break;
        
    #print(all_leaves)
    
    #print(Ij)
    
        
    ######################################Step-4##############################################
    for tree_leaves in all_leaves:
        #print("$$$$$$$$$$$$$$$$$$leave$$$$$$$$$$$$$$$$$$$$")
        for key in tree_leaves:
            #print("---------------DS----------------------")
            dj_indices = tree_leaves[key]
            #print(dj_indices)
            dj = df_incomplete_data.loc[dj_indices, :]
            dj.sort_index(inplace=True)
            if dj.isnull().sum().sum() != 0:
                
                # get record missed in dj
                            
                records_missed = dj[dj.isnull().any(axis=1)] #Xmis is the incomplete subset of X
                index_list_records_missed = records_missed.index
                
                for row in range(len(records_missed)):
                    
                    
                    index_of_row_missed = index_list_records_missed[row]
                                    
                    RI_Missed = records_missed.iloc[row]
                    
                    RI_Missed_CPY = RI_Missed.copy()
                    
                    
                    dj_cpy_without_RI_Ljl = dj.drop(index_list_records_missed)
                    
    
                    #########################################################
                    #Induce a random missing into xiz
                    #St.iloc[len(St.index)-1 , 20]=np.NaN 
                    random_missing_col = random.randint(0,len(dj_cpy_without_RI_Ljl.columns)-1)
                    AV = RI_Missed_CPY.iloc[random_missing_col]
                    
                    while(pd.isnull(AV)):
                        random_missing_col = random.randint(0,len(dj_cpy_without_RI_Ljl.columns)-1)
                        AV = RI_Missed_CPY.iloc[random_missing_col]
                        
                    #print("AV:",AV)
                    RI_Missed_CPY.iloc[random_missing_col]=np.NaN 
                    #########################################################
                        
                    mp = len(dj_cpy_without_RI_Ljl.index) # mp is the number of records in Pt
                    
                    K_List = []
                    RMSE_List = []
                    for k_count in range(1,mp+1):
                        K_List.append(k_count)
                        ###knni###
                        neighbors_for_xi_in_St = get_neighbors(dj_cpy_without_RI_Ljl.iloc[0:len(dj_cpy_without_RI_Ljl.index)].values.tolist(), RI_Missed_CPY.values.tolist(), k_count)
                        df_neighbors_for_xi_in_St = pd.DataFrame(data=neighbors_for_xi_in_St, columns=dj_cpy_without_RI_Ljl.columns)
                        
                        df_neighbors_for_xi_in_St = df_neighbors_for_xi_in_St.append(RI_Missed_CPY)
                        
                        #print(df_neighbors_for_xi_in_St)
                        
                        RI_filled_EMI = impy.em(df_neighbors_for_xi_in_St.values)
                        df_dI_after_EMI=pd.DataFrame(data=RI_filled_EMI, columns=df_neighbors_for_xi_in_St.columns, index=df_neighbors_for_xi_in_St.index)         
                
                        imputed_value = df_dI_after_EMI.iloc[len(df_dI_after_EMI.index)-1,random_missing_col]
                        
                        #print("imputed_value",imputed_value)
                        
                        actual_value = AV
                
                        average_RMSE_All_Cols_Final = np.sqrt(np.mean((imputed_value-actual_value)**2))
                        
                        
                        RMSE_List.append(average_RMSE_All_Cols_Final)
                        
                        ####
                       
                        #break;
                        
                        ######
                    
                    #print(K_List)
                    #print(RMSE_List)
                    index_of_min_value_in_RMSE_List = RMSE_List.index(min(RMSE_List))
                    
                    k_of_min_RMSE = K_List[index_of_min_value_in_RMSE_List]
                    #min_RMSE = RMSE_List[index_of_min_value_in_RMSE_List]
                    
                    #print(k_of_min_RMSE)
                    #print(min_RMSE)
                    
                    neighbors_for_Ri_in_Ljl_final = get_neighbors(dj_cpy_without_RI_Ljl.iloc[0:len(dj_cpy_without_RI_Ljl.index)].values.tolist(), RI_Missed.values.tolist(), k_of_min_RMSE)
                    df_neighbors_for_Ri_in_Ljl_final = pd.DataFrame(data=neighbors_for_Ri_in_Ljl_final, columns=dj_cpy_without_RI_Ljl.columns)
                        
                    Θi = df_neighbors_for_Ri_in_Ljl_final.append(RI_Missed)
                    
                    dj_filled_EMI = impy.em(Θi.values)
                    
                    df_dj_after_EMI=pd.DataFrame(data=dj_filled_EMI, columns=Θi.columns, index=Θi.index)
                    
                    
                    RI_imputed = df_dj_after_EMI.iloc[len(df_dj_after_EMI.index)-1]
            
                    #xi_imputed_with_index = pd.Series(RI_imputed).rename(index_of_row_missed)
                    
                    df_incomplete_data = df_incomplete_data.append(RI_imputed)
                
                    df_incomplete_data = df_incomplete_data.loc[~df_incomplete_data.index.duplicated(keep='last')]
                    
                    df_incomplete_data.sort_index(inplace=True)
                    
                    
                    dj = dj.append(RI_imputed)
                
                    dj = dj.loc[~dj.index.duplicated(keep='last')]
                    
                    dj.sort_index(inplace=True)
                    
                    #break
            #break

#print(df_incomplete_data)

    return df_incomplete_data

##########################################################################################


complete_data=pd.read_csv('Original-DataSet-Without-Missing.csv')
df_complete_data=pd.DataFrame(data=complete_data, columns=complete_data.columns, index=complete_data.index)

numvar= len(df_complete_data.columns)# num of columns


X_incomplete=pd.read_csv('1% #0- Dataset-With-Missing.csv')

df_incomplete_data=pd.DataFrame(data=X_incomplete, columns=X_incomplete.columns, index=X_incomplete.index)

start_time = time.time()

all_dataset_imputed = KDMI(complete_data , X_incomplete)

timeFinal = (time.time() - start_time)
print("--- %s seconds ---" % timeFinal)

all_dataset_imputed.to_csv("1% #8- Dataset-Without-Missing-After-KDMI.csv",index=False ,sep=';', encoding='utf-8')

print("^^^^^^^^^^^^^^^^^^RMSE^^^^^^^^^^^^^^^^^^")
sum_of_RMSE = 0
sum_of_NRMSE = 0
sum_of_AE = 0
for i in range(0,numvar):
	#print(i)
	#print("----------------------")
	l1_after_imputation = all_dataset_imputed.iloc[:,i].values
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

ws1['AZ6'] = average_RMSE_All_Cols
ws1['BA6'] = average_NRMSE_All_Cols
ws1['BB6'] = average_AE_All_Cols
ws1['BC6'] = timeFinal

wb.save(summaryName)


