# -*- coding: utf-8 -*-
"""
Created on Mon Nov 4 19:03:55 2019

@author: Mahmoud M. Ismail
"""

import pandas as pd
import random
import numpy

#import matplotlib.pyplot as plt
import missingno as msno 


def MNAR(my,start_row,start_col,mdp):
	
	numob= len(my.index) # num of rows
	#print(numob)
	
	numvar= len(my.columns)# num of columns
	#print(numvar)
	
	allcells=numob*numvar
	#############get MD Variables(cols)###############
	nV = 25 # change according to dataset , don't forget
	
	MDVariables = []
	for i in range(0,nV):
		randVar = random.randint(start_col, numvar - 1)
		while randVar in MDVariables:
			randVar = random.randint(start_col, numvar - 1)
			#print (randVar)

		MDVariables.append(randVar)
		#print (randVar)
		
	#print(MDVariables)
	
	############get MD observations(rows)#############
	observations = []
	numNullCells = 0
	while True:
		x = random.randint(start_row,numob-1)
		if x not in observations:
			observations.append(x)
			numNullCells += nV
			if((numNullCells/allcells)*100) > mdp:
				break
			
	#print(observations)
	
	######delete the data in the selected observations###### 
	#############for the selected variables#################
	
	for i in observations:
		for j in MDVariables:
			my.iloc[i,j]=numpy.NaN
	
	###################get MD percentage####################
	
	sum1=0
	for i in my.isnull().sum():
		sum1=sum1+i

	nullcells=sum1
    
	print("Missing data Percentage:",(nullcells/allcells)*100,"%")
	
	####get num of missing observations for each Variable####
    
	print("Number of Missing Values in each Column:")

	print(my.isnull().sum())	
	
	############create csv file with missing data############
			
	my.to_csv("1% #0- Dataset-With-Missing.csv",index=False ,sep=';', encoding='utf-8')
    
    #Visualize the number of missing 
	#values as a bar chart 
	fig_bar = my.isnull().sum().plot(kind='bar')
	fig_bar_copy = fig_bar.get_figure() 
	fig_bar_copy.savefig('1% #0-MNAR-Image-Bar.png')
	
	# Visualize the number of missing 
	# values as a bar chart 
	#This bar chart gives you an idea about how many missing values
	#are there in each column.
	#msno.bar(my) 
	
	# Visualize missing values as a matrix 
	#Using this matrix you can very quickly find the pattern of 
	#missingness in the dataset.
	fig_matrix = msno.matrix(my) 
	fig_matrix_copy = fig_matrix.get_figure() 
	fig_matrix_copy.savefig('1% #0-MNAR-Image-Matrix.png')
	
	
	
	# Visualize the correlation between the number of 
	# missing values in different columns as a heatmap 
	#msno.heatmap(my) 
	
	
	#The dendrogram allows you to more fully correlate
	#variable completion, revealing trends deeper
	#than the pairwise ones visible in the correlation heatmap:
	#msno.dendrogram(my)

	#my.plot.scatter(x='id', y='fk_univ_year', title='Iris Dataset')
    
    
	return my
	
	#########################################################
	
    
mdp = 1 # %
Original_DataSet_Without_Missing = pd.read_csv('Original-DataSet-Without-Missing.csv')
my=pd.read_csv('Original-DataSet-Without-Missing.csv')
my_with_missing = MNAR(my,0,0,mdp)


