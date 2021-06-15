import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#READ DATASET
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#TRAINING LINEAR REGRESSION MODEL ON WHOLE DATASET
from sklearn.linear_model import LinearRegression
model_linear_reg = LinearRegression()
model_linear_reg.fit(X, y)


#TRAINING POLYNOMIAL REGRESSION MODEL ON WHOLE DATASET
from sklearn.preprocessing import PolynomialFeatures
model_poly_reg = PolynomialFeatures(degree= 4)
X_poly = model_poly_reg.fit_transform(X)
model_linear_reg_2 = LinearRegression()
model_linear_reg_2.fit(X_poly, y)

'''
#LINEAR REGRESSION MODEL RESULTS
plt.scatter(X, y, color='red')
y_pred = model_linear_reg.predict(X)
plt.plot(X, y_pred, color='blue')
plt.title('Simple Linear Regression Salary Expectations')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
'''

'''
#POLYNOMIAL REGRESSION MODEL RESULTS
y_pred_poly = model_linear_reg_2.predict(X_poly)
plt.scatter(X, y, color='red')
plt.plot(X, y_pred_poly, color='blue')
plt.title('Poly Regression Model Salary Expectations')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
'''

'''
#POLYNOMIAL REGRESSION - BETTER VISUALIZATION RESULTS
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, model_linear_reg_2.predict(model_poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Poly Regression Model Salary Expectations')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
'''

#PREDICT VALUES WITH LINEAR REGRESSION
print(model_linear_reg.predict([[6.5]]))

#PREDICT VALUES WITH POLYNOMIAL REGRESSION
print(model_linear_reg_2.predict(model_poly_reg.fit_transform([[6.5]])))



