#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 22:35:44 2018

@author: macuser
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from math import sqrt
import matplotlib.pyplot as plt
#%matplotlib inline

#from customplot import *


# Create connection.
cnx = sqlite3.connect('database.sqlite')
cursor = cnx.cursor()
table_names = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

#Read sql tables into data frames to be analyzed.
df_player_att = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

print(df_player_att.columns)
print(df_player_att.describe().transpose())
#is any row NULL ?
print(df_player_att.isnull().any().any(), df_player_att.shape)
print(df_player_att.isnull().sum(axis=0))
# Fix it
# Take initial # of rows
rows = df_player_att.shape[0]
# Drop the NULL rows
df_player_att = df_player_att.dropna()
#Check if all NULLS are gone ?
print(rows)
print(df_player_att.isnull().any().any(), df_player_att.shape)
#How many rows with NULL values?
print(rows - df_player_att.shape[0])
#Shuffle the rows of df so we get a distributed sample when we display top few rows
df_player_att = df_player_att.reindex(np.random.permutation(df_player_att.index))

#--------------------------------------------------------------------------
# Part #2(a) - Player Attributes Table - Correlation - Players Overall Rating
#-------------------------------------------------------------------------

#Analyzing Player Attributes Table - Correlation

#Create a list of potential Features to measure correlation with
potentialFeatures = ['acceleration', 'curve', 'free_kick_accuracy', 'ball_control', 'shot_power', 'stamina']
#prints out the correlation coefficient of “overall_rating” of a player with each feature we added to the list as potential
# check how the features are correlated with the overall ratings
for f in potentialFeatures:
    related = df_player_att['overall_rating'].corr(df_player_att[f])
    print("%s: %f" % (f,related))
#selecting the columns and creating a list with correlation coefficients, called “correlations”.
cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes']
# create a list containing Pearson's correlation between 'overall_rating' with each column in cols
correlations = [ df_player_att['overall_rating'].corr(df_player_att[f]) for f in cols ]
#print(len(cols), len(correlations))

# create a function for plotting a dataframe with string columns and numeric values
def plot_dataframe(df_player_att, y_label):  
    color='coral'
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.ylabel(y_label)
    ax = df2.correlation.plot(linewidth=3.3, color=color)
    ax.set_xticks(df2.index)
    ax.set_xticklabels(df2.attributes, rotation=75); #Notice the ; (remove it and see what happens !)
    plt.show()    
# create a dataframe using cols and correlations
df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations}) 
# let's plot above dataframe using the function we created
plot_dataframe(df2, 'Player\'s Overall Rating')

#--------------------------------------------------------------------------
# Part #2(b) - Player Attributes Table - Grouping Players/Kmeans Clustering
#-------------------------------------------------------------------------
# Define the features to use for grouping players
select5features = ['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']
print(select5features)
# Generate a new dataframe by selecting the features you just defined
df_select = df_player_att[select5features].copy(deep=True)
print(df_select.head())

# KMeans to cluster the values (i.e., player features on gk_kicking, potential, marking, interceptions, and standing_tackle)
# Perform scaling on the dataframe containing the features
data = scale(df_select)
# Define number of clusters
noOfClusters = 4
# Train a model
model = KMeans(init='k-means++', n_clusters=noOfClusters, n_init=20).fit(data)
print(90*'_')
print("\nCount of players in each cluster")
print(90*'_')
print(pd.value_counts(model.labels_, sort=False))