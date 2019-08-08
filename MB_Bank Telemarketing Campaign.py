# -*- coding: utf-8 -*-
"""
ML Classification Project
Created on Fri Apr 14 22:49:24 2019

@author: mbigdelou
"""
#==== Import libraries
import os
os.chdir(r'C:\Users\mbigdelou\Desktop\Machine Learning Project')

os.getcwd()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm

plt.interactive(False)

#========================
#==== Import dataset
df = pd.read_csv('bank.csv', sep=",")

#General info about dataset
df.shape
df.describe()
df.head()

list(df)
df.info()

#checking if any value is missing
df.isnull().any()

#========================
#==== exploration of target variable and categorical variables
df.y.value_counts()
df.groupby('y').mean()
df.groupby('job').mean()
df.groupby('marital').mean()
df.groupby('education').mean()
df.groupby('month').mean()

#========================
#==== Preproccessing 
#==== Scale Dataset,  Normalization of Variables
#Since the ranges of features are similar, then there is no need to normalize data
#Transofming String variables to numerical and dummy variables
# month (since it follows an order, then it's possible to number them from 1 to 12)
def quantifylabelvar(input_labels, input_frame, column):
    for i in range(0,len(input_frame[column])):
        for j in range(0,len(input_labels)):
            if input_frame[column].values[i] == input_labels[j]:
                input_frame[column].values[i] = j
    return input_frame[column]

months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']  
df.month = quantifylabelvar(months, df, "month")
df.month+=1
del months

# job, marital, education 
# creating dummy variables
df_old = df.copy()
df = pd.get_dummies(df,columns=['job','marital','education'], drop_first=True, dummy_na=False)


#========================
#==== Correlation Matrix
correlation = df.corr()
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
plt.show()

#========================
#==== extracting independent and target variables
X = df.drop(['y'],axis=1)
y = df.y

#========================
#==== Spliting dataset into traing  and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=0) 

#========================
#==== Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors  import KNeighborsClassifier 
from sklearn.svm  import SVC
from sklearn.ensemble  import RandomForestClassifier, AdaBoostClassifier

logreg = LogisticRegression() 
knn = KNeighborsClassifier(n_neighbors=7, weights='distance') 
rfc = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=0) 
adbc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0) 
svmc = SVC(kernel='poly', degree=2, gamma='scale') 

#==== Train Regressor
logreg.fit(X_train,y_train)
knn.fit(X_train,y_train)
rfc.fit(X_train,y_train)
adbc.fit(X_train,y_train)
svmc.fit(X_train,y_train)

#==== Predict on the test set
y_pred_logreg = logreg.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_rfc = rfc.predict(X_test)
y_pred_adbc = adbc.predict(X_test)
y_pred_svmc = svmc.predict(X_test)

#==== Performance Measures
logreg.score(X_test,y_test)
knn.score(X_test,y_test)
rfc.score(X_test,y_test)
adbc.score(X_test,y_test)
svmc.score(X_test,y_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy_score(y_test,y_pred_logreg)
accuracy_score(y_test,y_pred_knn)
accuracy_score(y_test,y_pred_rfc)
accuracy_score(y_test,y_pred_adbc)
accuracy_score(y_test,y_pred_svmc)


confusion_matrix(y_test,y_pred_logreg)
confusion_matrix(y_test,y_pred_knn)
confusion_matrix(y_test,y_pred_rfc)
confusion_matrix(y_test,y_pred_adbc)
confusion_matrix(y_test,y_pred_svmc)

cr_logreg = classification_report(y_test,y_pred_logreg)
cr_knn = classification_report(y_test,y_pred_knn)
cr_rfc = classification_report(y_test,y_pred_rfc)
cr_adbc = classification_report(y_test,y_pred_adbc)
cr_svmc = classification_report(y_test,y_pred_svmc)


#==== K-Folds Cross Validation (6-fold cross validation)
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

scores_logreg = cross_val_score(logreg.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) logreg:', scores_logreg)

scores_knn = cross_val_score(knn.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) knn:', scores_knn)

scores_rfc = cross_val_score(rfc.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) rfc:', scores_rfc)

scores_adbc = cross_val_score(adbc.fit(X_train,y_train), X_train, y_train, cv=6)
print ('Cross-validated scores (cv=6) adbc:', scores_adbc)

'''#Cross validation not working on svmr
#scores_svmc = cross_val_score(svmc.fit(X_train,y_train), X_train, y_train, cv=6)
#print ('Cross-validated scores (cv=6) svmc:', scores_svmc)
''' 

#==== Grid Search hyper-parameter tuning 
from sklearn.model_selection import GridSearchCV
# Linear Regression does not need hyper-parameter tuning
# KNN
model_knn1 = KNeighborsClassifier() 

param_dict_knn = {
        'n_neighbors': [5,6,7,9,11], 
        'weights': ['uniform', 'distance'], 
        'leaf_size' : [10,20,25,30,35,40],
        }

model_knn2 = GridSearchCV(model_knn1,param_dict_knn)
model_knn2.fit(X_train,y_train)
model_knn2.best_params_
model_knn2.best_score_

# Random Forest Regressor
model_rfc1 = RandomForestClassifier() 

param_dict_rfc = {
        'n_estimators': [20,30,40,50,60], 
        'max_depth': [10,20,30,40,50],         
        }

model_rfc2 = GridSearchCV(model_rfc1, param_dict_rfc)
model_rfc2.fit(X_train,y_train)
model_rfc2.best_params_
model_rfc2.best_score_

# AdaBoost Regressor
model_adbc1 = AdaBoostClassifier()

param_dict_adbc = {
        'n_estimators': [30,40,50,60,70],        
        'learning_rate' : [.1,1,3,10],
        }

model_adbc2 = GridSearchCV(model_adbc1, param_dict_adbc)
model_adbc2.fit(X_train,y_train)
model_adbc2.best_params_
model_adbc2.best_score_

# SVC
model_svmc1 = SVC()

param_dict_svmc = {
        'gamma': ['auto'],
        'C' : [0.001,0.01,0.1,1,10],
        'kernel' : ['rbf', 'linear','poly', 'sigmoid'],        
        'degree' : [2,3,4,5]
        }

model_svmc2 = GridSearchCV(model_svmc1, param_dict_svmc, cv=None)
model_svmc2.fit(X_train,y_train)
model_svmc2.best_params_
model_svmc2.best_score_








