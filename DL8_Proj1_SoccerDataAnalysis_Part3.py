#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 22:46:13 2018

@author: macuser
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn import tree
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit
from sklearn.svm import SVC
from time import time
from math import sqrt
import matplotlib.pyplot as plt

 
# Create connection.
cnx = sqlite3.connect('database.sqlite')
cursor = cnx.cursor()
table_names = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

#Read sql tables into data frames to be analyzed.
#df_player_att = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
Player_Attributes = pd.read_sql_query("SELECT * from Player_Attributes", cnx)

#select relevant fields
Player_Attributes.dropna(inplace=True)
Player_Attributes.drop(['id', 'player_fifa_api_id', 'player_api_id', 'date'], axis = 1, inplace = True)
overall_rating = Player_Attributes['overall_rating']
features = Player_Attributes.drop('overall_rating', axis = 1)
features.head()

# Use LabelEncoder to convert categorical data field into numerical data field

le_sex = preprocessing.LabelEncoder()

#to convert into numbers

features.preferred_foot = le_sex.fit_transform(features.preferred_foot)
features.attacking_work_rate = le_sex.fit_transform(features.attacking_work_rate)
features.defensive_work_rate = le_sex.fit_transform(features.defensive_work_rate)
features.head()

# Use pandas get_dummies to convert categorical value into numerical
features = pd.get_dummies(features)

# Feature scaling using MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
scaled_features = min_max_scaler.fit_transform(features)

pca = PCA(n_components = 6)
pca_features = pca.fit_transform(scaled_features)

# Train and predict model on Decision tree and on SGD regressor

reg1 = tree.DecisionTreeClassifier()
reg2 = linear_model.SGDRegressor()
#reg3 = SVC(kernel='linear', C = 1.0)

#regs = {reg1:"Decision Tree", reg2:"SGDRegressor", reg3:"SVC"}
regs = {reg1:"Decision Tree", reg2:"SGDRegressor"}

for key in regs:
    t0 = time()
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, overall_rating, test_size=0.25, random_state=0)

    print ("--------------------")
    print (regs[key])
    print ("--------------------")

    t1 = time()
    key.fit(X_train, y_train)
    print ("Time taken to train the model: {}".format(time()-t1))

    t2 = time()
    pred_test = key.predict(X_test)
    pred_train = key.predict(X_train)
    print ("Time taken to predict the model: {}".format(time()-t2))

    t3 = time()
    print ("r2 score of this model on testing set is: {}".format(r2_score(y_test, pred_test)))
    print ("r2 score of this model on training set is: {}".format(r2_score(y_train, pred_train)))
    
# Use GridSearch to tune the model
def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)

    # TODO: Create a decision tree regressor object
       
    regressor1 = DecisionTreeRegressor()
    regressor2 = linear_model.SGDRegressor()
    #regressor3 = SVC()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    tree_params = {'max_depth' : [3, 6, 9, 20, 100], 'min_samples_split':[2, 3, 4, 5]}
    sgd_params = {'loss':['squared_loss', 'huber'], 'penalty': ['none', 'l2', 'l1', 'elasticnet'], 'n_iter':[10, 75, 100, 500]}
   #svm_params = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
    
    
    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # Updated cv_sets and scoring parameter
    grid = GridSearchCV(regressor1, tree_params, scoring = scoring_fnc, cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    #print("grid fit")
    grid = grid.fit(X, y)

    # Updated cv_sets and scoring parameter
    #grid = GridSearchCV(regressor2, sgd_params, scoring = scoring_fnc, cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    #grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def performance_metric(y_true, y_predict):
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'

    
    score = r2_score(y_true, y_predict)
    # Return the score
    return score

print("Script Start")
t0 = time()
grid_reg = fit_model(pca_features, overall_rating)
print (grid_reg.score)
# grid_pred = grid_reg()
print ("Time taken to train and predict using GridSearch: {}".format(time() - t0))
print ("Best parameters are: {}".format(grid_reg.get_params()))