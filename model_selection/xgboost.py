#Best for Regression and Classfication

#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#IMPORT DATASET
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#SPLIT DATA - TRAINING AND TEST 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#TRAIN XBOOST ON TRAINING SET
from xgboost import XGBClassifier #Change this to XGBRegressor for regression
model = XGBClassifier()
model.fit(X_train, y_train)

#CONFUSION MATRIX 
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


#APPLY K-FOLD CROSS VALIDATION
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))