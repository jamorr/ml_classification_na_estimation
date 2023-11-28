#
from __future__ import division 

from pytesmo.metrics import rmsd
from pytesmo.metrics import nrmsd
from pytesmo.metrics import aad
#

import math

from math import sqrt

import numpy as np

import random

#from ycimpute.imputer import EM
import pandas as pd

import time

import impyute as impy

from fancyimpute import SoftImpute
from fancyimpute import IterativeImputer

from numpy import linalg as LA

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

    
# =============================================================================
# #############################################################################
# =============================================================================
def DempsterShafer(yq_list , n):
    
    data = np.array(yq_list)

    shape = ( n, n )
    
    J1n = np.ones((1,n))
    
    
    Γ  = data.reshape( shape )
    
    ϕ = Γ.copy()
    
    ψ = Γ.mean(0)
    
    for i in range(n):
        γi = Γ[i]
        
        total_bottom_part = 0
        for jt in range(n):
            matrix_norm = LA.norm(np.subtract((ψ[jt] * J1n) , (γi) ))
            top_part_t = math.pow((1 + matrix_norm), -1)
            total_bottom_part = total_bottom_part + top_part_t
            
        for j in range(n):
            matrix_norm = LA.norm(np.subtract((ψ[j] * J1n) , (γi) ))
            top_part = math.pow((1 + matrix_norm), -1)
            ϕ[j][i] = top_part / total_bottom_part
          
        #print(Γ[i][j])
    
    ########////////////////////////////////////////////////########
    
    β = ϕ.copy()
    
    for i in range(n):
        for j in range(n):
            top_part_mutiply_total = 1
            for k in range(n):
                if k != j:
                    top_part_mutiply_total = top_part_mutiply_total *  (1 - ϕ[k][i])
            
            top_part =  (ϕ[j][i]) * top_part_mutiply_total
            bottom_part = 1 - ((ϕ[j][i])*(1 - top_part_mutiply_total))
            β[i][j] = top_part /  bottom_part
      
    ########////////////////////////////////////////////////########
    λ = []
    for j in range(n):
        mutiply_total = 1
        for i in range(n):
            mutiply_total = mutiply_total * β[i][j]
        λ.append(mutiply_total)
    ########////////////////////////////////////////////////########
    
    Λ = β.copy()
    
    for j in range(n):
        for i in range(n):
            Λ[i][j] = λ[j]
    
    
    γfinal = ( (LA.norm(np.add(Γ,Λ))) + (LA.norm(np.subtract(Γ,Λ))) ) / math.factorial(n)
    
    #print(γfinal)
    
    return γfinal
    
# =============================================================================
# #############################################################################
# =============================================================================
    
def kEMI_PLUS(complete_data , X_incomplete):
    #complete_data=pd.read_csv('Sonar_Without_Missing.csv')
    
    #X_incomplete=pd.read_csv('Sonar_With_Missing_MCAR.csv')
        
    #df_complete_data=pd.DataFrame(data=complete_data, columns=complete_data.columns, index=complete_data.index)
      
    #numvar= len(df_complete_data.columns)# num of columns
    
    df_incomplete_data=pd.DataFrame(data=X_incomplete, columns=X_incomplete.columns, index=X_incomplete.index)
    
    #Split X into Xobs and Xmis subsets
    
    Xmis = df_incomplete_data[df_incomplete_data.isnull().any(axis=1)] #Xmis is the incomplete subset of X
    
    #print(Xmis.shape)
    
    Xobs = df_incomplete_data.dropna(inplace=False) #Xobs is the complete subset of X
    
    #print(Xobs.shape)
    
    Pt = Xobs #Pt is the pool that is initialized to Xobs
    
    while not(Xmis.empty):
        #print("------------------------------------------------------------------------")
        # Select an incomplete record from Xmis and add it into the pool
        xi = Xmis.iloc[0]
        
        #get index of xi to put the imputed xi in correct order in dataset
        index_of_xi = Xmis.index.tolist()[0]
    
        #print(xi)
        Pt = Pt.append(xi)
            
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
            
            imputed_value = one_col_df_neighbors_for_xi_in_St.mean()[0]
            
            actual_value = St_Complete_Temp.iloc[len(St.index)-1 , random_missing_col]
    
            average_RMSE_All_Cols_Final = np.sqrt(np.mean((imputed_value-actual_value)**2))
            
            RMSE_List.append(average_RMSE_All_Cols_Final)
            
            #break
            
            ######
        print("K_List",K_List)
        #print(RMSE_List)
        
        print("-----------------------------")
        K_List_Sorted = [x for _,x in sorted(zip(RMSE_List,K_List))] # final
        print(K_List_Sorted) 
        
        RMSE_List_sorted =  sorted(RMSE_List) # final
        print(RMSE_List_sorted) 
        
        Q = []
        
        for i in range(1,mp) :
            if (i*i) <= (mp - 1):
                Q.append(i)
            else:
                break
        
        n = max(Q)
        
        print(Q)
        ######### get indices of missing values #########
        indices_missing_in_xi = []
        for i in range(len(xi)):
            if pd.isnull(xi[i]):
                indices_missing_in_xi.append(i)
        print("missing cols:", indices_missing_in_xi)
        #################################################
        
        
        γq_DF_LIST = list()
        
        yq_for_multi_rows_multi_cols_after_imput  = []
        
        print("size: ",len(K_List_Sorted))
        for q in range(1,(n*n)+1):
            kq = K_List_Sorted[0] # Find the lowest RMSE and return corresponding kq
            K_List_Sorted.pop(0) # Update K_List_Sorted
            RMSE_List_sorted.pop(0) # Update RMSE_List_sorted
            
            neighbors_αi = get_neighbors(Pt.iloc[0:len(Pt.index)-1].values.tolist(), xi.values.tolist(), kq)
            df_neighbors_αi = pd.DataFrame(data=neighbors_αi, columns=Pt.columns)
            
            Θi = df_neighbors_αi.append(xi)
            
            ####EMI####
            Θi_filled_EM = impy.em(Θi.values)
            #Θi_filled_EM = SoftImpute().fit_transform(Θi.values)
            #Θi_filled_EM = IterativeImputer().fit_transform(Θi.values)
            ###########
            
            df_Θi_filled_EM=pd.DataFrame(data=Θi_filled_EM, columns=Pt.columns)
            
            xi_imputed = df_Θi_filled_EM.iloc[len(df_Θi_filled_EM.index)-1]
            
            xi_imputed_with_index = pd.Series(xi_imputed).rename(index_of_xi)
            
            γq_DF_LIST.append(xi_imputed_with_index)
            
            yq_for_one_row_multi_cols_after_imput = []
            
            for index in indices_missing_in_xi:
                yq_for_one_row_multi_cols_after_imput.append(xi_imputed_with_index[index])
            
            yq_for_multi_rows_multi_cols_after_imput.append(yq_for_one_row_multi_cols_after_imput)
            
            #break
            
        for j in range(len(yq_for_multi_rows_multi_cols_after_imput[0])):
            yq_list = []
            for i in range(len(yq_for_multi_rows_multi_cols_after_imput)):
                yq = yq_for_multi_rows_multi_cols_after_imput[i][j]
                yq_list.append(yq)
            
            γfinal = DempsterShafer(yq_list , n)
                    
            xi[indices_missing_in_xi[j]] = γfinal
            
            #break
        
        xi_imputed_with_index = pd.Series(xi).rename(index_of_xi)
        Xobs = Xobs.append(xi_imputed_with_index)
            
        Xmis = Xmis.iloc[1:]
        
        Pt = Xobs
        
        #break
    
    Xobs.sort_index(inplace=True)
    all_dataset_imputed = Xobs
  
    return all_dataset_imputed

##############################################

complete_data=pd.read_csv('Original-DataSet-Without-Missing.csv')
df_complete_data=pd.DataFrame(data=complete_data, columns=complete_data.columns, index=complete_data.index)

numvar= len(df_complete_data.columns)# num of columns


X_incomplete=pd.read_csv('1% #0- Dataset-With-Missing.csv')
df_incomplete_data=pd.DataFrame(data=X_incomplete, columns=X_incomplete.columns, index=X_incomplete.index)

start_time = time.time()

all_dataset_imputed = kEMI_PLUS(complete_data , X_incomplete)

timeFinal = (time.time() - start_time)
print("--- %s seconds ---" % timeFinal)

all_dataset_imputed.to_csv("1% #10- Dataset-Without-Missing-After-KEMI_PLUS.csv",index=False ,sep=';', encoding='utf-8')

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

ws1['BH6'] = average_RMSE_All_Cols
ws1['BI6'] = average_NRMSE_All_Cols
ws1['BJ6'] = average_AE_All_Cols
ws1['BK6'] = timeFinal

wb.save(summaryName)


