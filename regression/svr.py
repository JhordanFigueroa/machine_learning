import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#READ DATASET
dataset = pd.read_csv('Position_Salaries.csv')
#X - features - dependent variable
X = dataset.iloc[:, 1:-1].values
#y - wanting to predict - indendent variable
y = dataset.iloc[:, -1].values

#RESHAPE y - row, column
y = y.reshape(len(y),1)

#FEATURE SCALING
'''
Don't Apply feature scaling - onehotencoding when having caterorical data (dummy variables) and need 1 or 0.
Don't Apply feature scaling - when dependent variable are binary, 0 or 1. 
Apply feature scaling - when one variable takes in super high variables compared to other variables - model will ignore lower 
numbers. 
Apply feature scaling - when have to split up data into training set and test set - apply feature scaling after split 
Sklearn Standard Scalar class expects the same type of data type - 2D array 
'''
from sklearn.preprocessing import StandardScaler
#Need two different transforms for X and y because of different means withing the data - scaling won't be the same if done together - want Ranges for values - 0-3 
X_sc = StandardScaler()
y_sc = StandardScaler()
X = X_sc.fit_transform(X)
y = y_sc.fit_transform(y)

'''
print(X)
print('*****')
print(y)
'''

#TRAINING THE SVR MODEL ON THE WHOLE DATASET
from sklearn.svm import SVR
#Kernel is how model learns - type of model - 
model_reg = SVR(kernel='rbf')
model_reg.fit(X, y)

#PREDICTING A NEW RESULT
#Need to transform values into orignial values because model was scaled 
pred = y_sc.inverse_transform(model_reg.predict(X_sc.transform([[6.5]])))
#print(pred)

#VISUALIZE SVR RESULTS
plt.scatter(X_sc.inverse_transform(X), y_sc.inverse_transform(y), color = 'red')
plt.plot(X_sc.inverse_transform(X), y_sc.inverse_transform(model_reg.predict(X)), color = 'blue')
plt.title('SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#VISUALIZE SVR RESULTS WITH BETTER RESOLUTION 
