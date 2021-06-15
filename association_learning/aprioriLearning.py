import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORT DATASET
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

#APYORI - has a list of values - not a dataframe  - need lists of list - elements need to be strings 
transcations = []
for i in range(0, len(dataset)):
    transcations.append([str(dataset.values[i,j]) for j in range(0,20)])

#print(transcations)

#Apriori
from apyori import apriori

#Model - min_support - the rules you want as output - what you want to get out of data
rules = apriori(transactions= transcations, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

results = list(rules)
#print(results)

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

#print(resultsinDataFrame)

#SORT DATAFRAME BY COLUMN IN DESCENDING 
highestLift = resultsinDataFrame.nlargest(n = 10, columns='Lift')
print(highestLift)