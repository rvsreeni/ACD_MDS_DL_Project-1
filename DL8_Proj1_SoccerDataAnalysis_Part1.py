#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 12:17:04 2018

@author: macuser
"""
import sys
import sqlite3
import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt

#-----------------------------------------
# Part #1(a) - Data Ingestion/Exploration
#-----------------------------------------

# Create connection.
cnx = sqlite3.connect('database.sqlite')
cursor = cnx.cursor()
table_names = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

#Read all sql tables into data frames to be analyzed.
df_player = pd.read_sql_query("SELECT * FROM Player", cnx)
df_player_att = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df_match = pd.read_sql_query("SELECT * FROM Match", cnx)
df_league = pd.read_sql_query("SELECT * FROM League", cnx)
df_country = pd.read_sql_query("SELECT * FROM Country", cnx)
df_team = pd.read_sql_query("SELECT * FROM Team", cnx)
df_team_att = pd.read_sql_query("SELECT * FROM Team_Attributes", cnx)

#Analyzing Country Table
print(90*"-")
print("Country Table")
print(df_country.describe())
print(90*"-")
print(df_country.isnull().sum(axis=0))
print(90*"-")
print(df_country)

#Analyzing League Table
print(90*"-")
print("League Table")
print(df_league.describe())
print(90*"-")
print(df_league.isnull().sum(axis=0))
print(90*"-")
print(df_league)

#Analyzing Team Table
print(90*"-")
print(df_team.describe())
print(90*"-")
print(df_team.isnull().sum(axis=0))
print(df_team[df_team.loc[:,'team_fifa_api_id'].isnull()])
df_team_updated = df_team[~df_team.loc[:,'team_fifa_api_id'].isnull()]
my_team = dict()
for i,j in list(df_team_updated.iloc[:,3:].groupby('team_short_name')):
    my_team[i] = j.iloc[:,0].values.tolist()
#List of teams with similar short team names 
print("List of teams with similar short team names") 
print("List of teams with similar short team names") 
print("List of teams with similar short team names")    
print({k:v for k,v in my_team.items() if len(v) > 1})
   
#Analyzing Team Attributes Table
print(90*"-")
print(df_team_att.describe())
print(90*"-")
print(df_team_att.isnull().sum(axis=0))

df_team_att_updated1 = df_team_att.drop(['buildUpPlayDribbling'],axis = 1)

tat = df_team_att_updated1.loc[:,df_team_att_updated1.columns.tolist()[3:]]

#Analyzing Player Table
print(90*"-")
print(df_player.describe())
print(90*"-")
print(df_player.isnull().sum(axis=0))

print("Cardinality of Feature: Height - {:0.3f}%".format( \
        100 * (len(np.unique(df_player.loc[:,'height'])) / len(df_player.loc[:,'height']))))
print("Cardinality of Feature: Weight - {:0.3f}%".format( \
        100 * (len(np.unique(df_player.loc[:,'weight'])) / len(df_player.loc[:,'weight']))))

#Analyzing Player Attributes Table
print(90*"-")
print(df_player_att.describe())
print(90*"-")
print(df_player_att.isnull().sum(axis=0))

print(np.unique(df_player_att.dtypes.values))

#print(df_player_att.select_dtypes(include =['float64','int64']).head().\
#loc[:,df_player_att.select_dtypes(include =['float64','int64']).columns[3:]].head())

corr2 = df_player_att.select_dtypes(include =['float64','int64']).\
loc[:,df_player_att.select_dtypes(include =['float64','int64']).columns[3:]].corr()

print(df_player_att['attacking_work_rate'].value_counts())
print(100*'*')
print(df_player_att['defensive_work_rate'].value_counts())
print(100*'*')
#print(df_player_att.shape)

df_player_att.loc[~(df_player_att['attacking_work_rate'].\
                                                  isin(['medium','high','low'])\
                       | df_player_att['defensive_work_rate'].isin(['medium','high','low'])),:].head()
df_player_att_updated1 = df_player_att.loc[(df_player_att['attacking_work_rate'].\
                                                  isin(['medium','high','low'])\
                       & df_player_att['defensive_work_rate'].isin(['medium','high','low'])),:]
#print(df_player_att_updated1.shape)
#print(df_player_att_updated1.head())

att_work_rate = df_player_att_updated1.groupby('attacking_work_rate').size().values.tolist()
def_work_rate = df_player_att_updated1.groupby('defensive_work_rate').size().values.tolist()

print("Attacking work rate factor, Medium, accounts for: {:0.3f}% of features".format(100 * att_work_rate[2]/np.sum(att_work_rate)))
print("Defensive work rate factor, Medium, accounts for: {:0.3f}% of features".format(100 * def_work_rate[2]/np.sum(def_work_rate)))

pat = df_player_att_updated1.loc[:,df_player_att_updated1.columns.tolist()[3:]]

#---------------------------------
# Part #1(b) - Data Visualization
#---------------------------------

# Player Table - Features (Height, Weight) - Scatter Plot

fig1, ax1 = plt.subplots(nrows = 1, ncols = 2)
fig1.set_size_inches(14,4)
sns.boxplot(data = df_player.loc[:,["height",'weight']], ax = ax1[0])
ax1[0].set_xlabel('Player Table Features')
ax1[0].set_ylabel('')
sns.distplot(a = df_player.loc[:,["height"]], bins= 10, kde = True, ax = ax1[1], \
            label = 'Height')
sns.distplot(a = df_player.loc[:,["weight"]], bins= 10, kde = True, ax = ax1[1], \
            label = 'Weight')
ax1[1].legend()
sns.jointplot(x='height',y = 'weight',data = df_player,kind = 'scatter')
fig1.tight_layout()

# Player Attribute Table - Correlation (ontinuous features) - Heat Map

# Correlation between the continuous features. We should see a positive correlation 
# between the attacking features, a positive correlation between the defensive features 
# and a negative correlation between the attacking and defensive features

fig2,ax2 = plt.subplots(nrows = 1,ncols = 1)
fig2.set_size_inches(w=24,h=24)
sns.heatmap(corr2,annot = True,linewidths=0.5,ax = ax2)

# Player Attribute Table - Count Plot
# Features - Preferred Foot, Attacking Work Rate, Defensive Work Rate

fig3, ax3 = plt.subplots(nrows = 1, ncols = 3)
fig3.set_size_inches(12,4)
sns.countplot(x = df_player_att['preferred_foot'],ax = ax3[0])
sns.countplot(x = df_player_att['attacking_work_rate'],ax = ax3[1])
sns.countplot(x = df_player_att['defensive_work_rate'],ax = ax3[2])
fig3.tight_layout()

fig4, ax4 = plt.subplots(nrows = 1, ncols = 3)
fig4.set_size_inches(12,3)
sns.countplot(x = df_player_att_updated1['preferred_foot'],ax = ax4[0])
sns.countplot(x = df_player_att_updated1['attacking_work_rate'],ax = ax4[1])
sns.countplot(x = df_player_att_updated1['defensive_work_rate'],ax = ax4[2])
fig4.tight_layout()

# Player Attribute Table - Bar Plot (Percentage)
# Features - Preferred Foot, Attacking Work Rate, Defensive Work Rate

fig4, ax4 = plt.subplots(nrows = 1, ncols = 3)
fig4.set_size_inches(12,3)
sns.barplot(x ='preferred_foot', y = 'preferred_foot', data = df_player_att_updated1,\
            estimator = lambda x: len(x)/len(df_player_att_updated1) * 100, ax = ax4[0],\
           orient = 'v')
ax4[0].set(ylabel = 'Percentage',title = 'Preferred Foot')
sns.barplot(x ='attacking_work_rate', y = 'attacking_work_rate', data = df_player_att_updated1,\
            estimator = lambda x: len(x)/len(df_player_att_updated1) * 100, ax = ax4[1],\
           orient = 'v')
ax4[1].set(ylabel = 'Percentage',title = 'Attacking Work Rate')
sns.barplot(x ='defensive_work_rate', y = 'defensive_work_rate', data = df_player_att_updated1,\
            estimator = lambda x: len(x)/len(df_player_att_updated1) * 100, ax = ax4[2],\
           orient = 'v')
ax4[2].set(ylabel = 'Percentage',title = 'Defensive Work Rate')
fig4.tight_layout()

# Player Attribute Table - Distribution/Spread of continuous features based off of categorical features

fig5, ax5 = plt.subplots(nrows=5,ncols=7)
fig5.set_size_inches(16,12)
for i,j in enumerate(df_player_att_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.distplot(pat.loc[:,j],kde = False,hist = True, ax = ax5[int(i/7)][i%7])
fig5.tight_layout()

# Player Attribute Table - Box Plot - Preferred Foot Vs Continuous Features

fig6, ax6 = plt.subplots(nrows=5,ncols=7)
fig6.set_size_inches(16,12)
for i,j in enumerate(df_player_att_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.boxplot(x = "preferred_foot", y = j, data= pat, ax = ax6[int(i/7)][i%7])
fig6.tight_layout()

# Player Attribute Table - Box Plot - Attacking Work Rate Vs Continuous Features

fig7, ax7 = plt.subplots(nrows=5,ncols=7)
fig7.set_size_inches(16,12)
for i,j in enumerate(df_player_att_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.boxplot(x = "attacking_work_rate", y = j, data= pat, ax = ax7[int(i/7)][i%7])
fig7.tight_layout()

# Player Attribute Table - Box Plot - Defensive Work Rate Vs Continuous Features

fig8, ax8 = plt.subplots(nrows=5,ncols=7)
fig8.set_size_inches(16,12)
for i,j in enumerate(df_player_att_updated1.select_dtypes(include = ['float64','int64']).columns[3:].tolist()):
    sns.boxplot(x = "defensive_work_rate", y = j, data= pat, ax = ax8[int(i/7)][i%7])
fig8.tight_layout()

# Team Attribute Table - Correlation of continuous features
sns.pairplot(tat)

# Team Attribute Table - Distribution/Spread of continuous features

fig9, ax9 = plt.subplots(nrows=2,ncols=4)
fig9.set_size_inches(12,6)
for i,j in enumerate(df_team_att_updated1.select_dtypes(include = ['int64']).columns[3:].tolist()):
    sns.distplot(tat.loc[:,j],kde =True,hist = True, ax = ax9[int(i/4)][i%4])
fig9.tight_layout()

df_team_att_updated1.select_dtypes(include = ['int64']).head()

#sns.boxplot(data = df_team_att_updated1.select_dtypes(include = ['int64']).iloc[:,3:],\
#           orient = 'h')
# 
#fig9, ax9 = plt.subplots(nrows=3,ncols=4)
#fig9.set_size_inches(14,8)
#for i,j in enumerate(df_team_att_updated1.select_dtypes(include = ['object']).columns[1:].tolist()):
#    #sns.countplot(tat.loc[:,j], ax = ax9[int(i/4)][i%4])
#    sns.barplot(x = j, y = j, data = tat,\
#            estimator = lambda x: len(x)/len(tat) * 100, ax = ax9[int(i/4)][i%4],\
#           orient = 'v')
#    ax9[int(i/4)][i%4].set(xlabel = "")
#fig9.tight_layout()

tat.select_dtypes(include = ['int64']).columns.tolist()

#sns.pairplot(tat,hue = tat.select_dtypes(include = ['object']).\
#          columns.tolist()[1]) 
#
#sns.pairplot(tat,hue = tat.select_dtypes(include = ['object']).\
#          columns.tolist()[12]) 

# Team Attribute Table - Build up play speed Vs Remaining features

fig9, ax9 = plt.subplots(nrows=2,ncols=4)
fig9.set_size_inches(12,6)
for i,j in enumerate(df_team_att_updated1.select_dtypes(include = ['int64']).columns[3:].tolist()):
    sns.boxplot(data = tat, y = j, x = tat.select_dtypes(include = ['object']).columns[3],\
                                                      ax = ax9[int(i/4)][i%4])
fig9.tight_layout()

