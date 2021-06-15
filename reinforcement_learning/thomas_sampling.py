import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
import random

#THOMAS SAMPLING - BETTER THAN UPPER CONFIDENCE INTERVAL. CAN HAVE DELAYED FEEDBACK FOR UPDATED DATA 

#IMPORT DATASET 
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#SET VARIABLES 
N = 250 #Goal is to find the least number of rounds needed to find the best ad - Change to 1000, 10000 originally - Number of datapoints
num_ads = 10
ads_selected = []
numbers_rewards_1 = [0]*num_ads
numbers_rewards_0 = [0]*num_ads
total_rewards = 0

#THOMAS SAMPLING
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, num_ads):
        random_beta = random.betavariate(numbers_rewards_1[i] + 1, numbers_rewards_0[i] + 1)
        if (random_beta > max_random): #GET MAXIMUM RANDOM 
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_rewards_1[ad] = numbers_rewards_1[ad] + 1
    else:
        numbers_rewards_0[ad] = numbers_rewards_0[ad] + 1
    total_rewards = total_rewards + reward


#VISUALIZE RESULTS 
plt.hist(ads_selected)
plt.title("Hist - Ad Selections")
plt.xlabel('Ads')
plt.ylabel('Number of times Ad Selected')
plt.show()

