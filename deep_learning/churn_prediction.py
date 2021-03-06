# -*- coding: utf-8 -*-
"""Churn_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RlpP_gt02LbkeEQv2xaDD0ZsY_N78b4_
"""

from google.colab import drive
drive.mount('/content/drive')

cd '/content/drive/MyDrive/ML/datasets'

"""DATA PROCESSING

"""

import pandas as pd
import numpy as np
import tensorflow as tf

tf.__version__

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

print(X)

#ENCODING CATEGORICAL DATA - COUNTRY AND GENDER DATA 
#LABELING GENDER
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:,2])

print(X)

#Geography Column - One Hot Encoding - because more than 2 variables 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

#SPLIT DATASET INTO TEST AND TRAINING SETS
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#FEATURE SCALING - NECESSARY FOR DEEP LEARNING AND ANN - APPLY TO ALL FEATURES IN TRAINING/TEST DATA- SCALE 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

"""BUILDING THE ANN"""

#INITIALIZE THE ANN
ann = tf.keras.models.Sequential()

#ADDING INPUT LAYER AND THE FIRST HIDDEN LAYER
ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))

#ADDING SECOND LAYER
#Activiation Function - Rectifier Function
ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))

#ADDING OUTPUT LAYER 
#This depends on dependent variable (y) and data of how its encoded (binary or not)
#Sigmoid Activation Function - works better for binary outcome 
#If dealing with output data as 3 or more - need to use SUFTMAX 
ann.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

"""TRAINING THE ANN"""

#COMPILING ANN
#NON BINARY - CROSS_ENTROPY 
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

#TRAINING ANN ON TRAINING SET
#Batch more efficient - several predictions learning
#Epochs - how many occurances of training required 
ann.fit(X_train, y_train, batch_size=32, epochs=100)

"""MAKING PREDICTIONS AND EVALUATING MODEL

Use our ANN model to predict if the customer with the following informations will leave the bank:

Geography: France

Credit Score: 600

Gender: Male

Age: 40 years old

Tenure: 3 years

Balance: $ 60000

Number of Products: 2

Does this customer have a credit card ? Yes

Is this customer an Active Member: Yes

Estimated Salary: $ 50000

So, should we say goodbye to that customer ?
"""

#PREDICT IF CUSTOMER STAYS OR LEAVES
#Predict method needs double square brackett always 
#Predict must have the same scaling was applied to the model
#New observations - transform method
print(ann.predict(sc.transform([[1, 0 , 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))) #Gives probability 
print(ann.predict(sc.transform([[1, 0 , 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

#PREDICTING TEST RESULTS
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test),1)),1))

#ANALYZE MODEL
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test, y_pred)