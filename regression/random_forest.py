import numpy as np
from numpy.core.defchararray import mod
import pandas as pd
import matplotlib.pyplot as plt

#READ DATASET
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values #Check for higher dimensions
y = dataset.iloc[:, -1].values

#print(X)

#Check for Encoding Categorical Data

#Decision Tree and Regression - Don't need to do any feature scaling 

#Training The Decision Tree Regression Model On All Dataset
from sklearn.ensemble import RandomForestRegressor
model_random_forest_reg = RandomForestRegressor(n_estimators=10, random_state=0)
model_random_forest_reg.fit(X, y)

#Predicting New Result
y_pred = model_random_forest_reg.predict([[6.5]])
print(y_pred)

#Visualize Results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, model_random_forest_reg.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()






