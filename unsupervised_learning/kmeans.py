import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#IMPORT DATASET
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

