from os import lseek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#print(X)
#print('****')
#print(y)


#SPLITTING THE DATASET INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/3, random_state= 0)

#TRAIN MODEL ON TRAINING SET
from sklearn.linear_model import LinearRegression
regressor_model = LinearRegression()
regressor_model.fit(X_train, y_train)

#PREDICT THE TEST SET RESULTS
y_pred = regressor_model.predict(X_test)
#print(y_pred)

'''
#VISUALIZE TRAINING SET RESULTS
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor_model.predict(X_train), color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
'''

#VISUALIZE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor_model.predict(X_train), color='blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#plt.show()

#MAKE SINGLE PREDICTION
print(regressor_model.predict([[12]]))

#PREDICTED LINEAR REGRESSION EQUATION LINE
print(regressor_model.coef_)
print(regressor_model.intercept_)


#Analyze Model
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

