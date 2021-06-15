#MULTI-ARM BANDIT PROBLEM - combine exploration and explotation 
#Law of Large Numbers 
#Each column has a fixed conversion rate - usual assumption - ex: ad will convert certain % of users over time 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 

#IMPORT DATASET 
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#SET VARIABLES 
N = 1000 #Goal is to find the least number of rounds needed to find the best ad - Change to 1000, 10000 originally - Number of datapoints
num_ads = 10
ads_selected = []
number_of_selections = [0]*num_ads
sums_of_rewards = [0]*num_ads
total_reward = 0

#ALGORITHM

for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, num_ads):
        if number_of_selections[i] > 0:
            avg_reward = sums_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/number_of_selections[i])
            upper_bound = avg_reward+ delta_i
        else: 
            upper_bound = 1e400 #Need high upper bound for algo
        if upper_bound > max_upper_bound: 
            max_upper_bound = upper_bound
            ad = i 
    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

#VISUALIZE RESULTS 
plt.hist(ads_selected)
plt.title("Hist - Ad Selections")
plt.xlabel('Ads')
plt.ylabel('Number of times Ad Selected')
plt.show()