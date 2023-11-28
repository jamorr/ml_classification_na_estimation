# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 23:07:41 2019

@author: Mahmoud M. Ismail
"""

import pandas as pd
import random
import numpy
#import matplotlib.pyplot as plt
import missingno as msno 



def MCAR(my,start_row,start_col,mdp):
	
	numob= len(my.index) # num of rows
	#print("num of rows:", numob)
	
	numvar= len(my.columns)# num of columns
	#print("num of columns:", numvar)
	
	allcells=numob*numvar
	
	while True:
		x=random.randint(start_row,numob-1)
		y=random.randint(start_col,numvar-1)
		if my.iloc[x,y]!=numpy.NaN:
			my.iloc[x,y]=numpy.NaN
			sum1=0
			for i in my.isnull().sum():
				sum1=sum1+i
			#print(sum1)
			nullcells=sum1
			if((nullcells/allcells)*100)>mdp:
				break
	sum1=0
	for i in my.isnull().sum():
		sum1=sum1+i

	nullcells=sum1

	print("Missing data Percentage:",(nullcells/allcells)*100,"%")
	
	print("Number of Missing Values in each Column:")
	print(my.isnull().sum())	
	
	my.to_csv("1% #0- Dataset-With-Missing.csv",index=False ,sep=';', encoding='utf-8')
	
	#Visualize the number of missing 
	#values as a bar chart 
	fig_bar = my.isnull().sum().plot(kind='bar')
	fig_bar_copy = fig_bar.get_figure() 
	fig_bar_copy.savefig('1% #0-MCAR-Image-Bar.png')

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
	fig_matrix_copy.savefig('1% #0-MCAR-Image-Matrix.png')
	
	
	
	# Visualize the correlation between the number of 
	# missing values in different columns as a heatmap 
	#msno.heatmap(my) 
	
	
	#The dendrogram allows you to more fully correlate
	#variable completion, revealing trends deeper
	#than the pairwise ones visible in the correlation heatmap:
	#msno.dendrogram(my)

	#my.plot.scatter(x='id', y='fk_univ_year', title='Iris Dataset')
    
	return my

mdp = 1 # %
Original_DataSet_Without_Missing = pd.read_csv('Original-DataSet-Without-Missing.csv')
my=pd.read_csv('Original-DataSet-Without-Missing.csv')
my_with_missing = MCAR(my,0,0,mdp)





