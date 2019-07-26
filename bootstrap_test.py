# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 20:25:58 2019

@author: Vlad
"""

#Bootstrapping test with linear regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Showing that bootstrap confidence intervals are similar to ones
#calculate using traditional formula
X = np.random.randn(10000)
N = 100

lower_intervals = []
upper_intervals = []
sample_means = []
for i in range(len(X) // N):
    sample = np.random.choice(X, size=N)
    mean = np.mean(sample)
    sample_means.append(mean)
    
st_dev = np.std(sample_means)

for i in sample_means:
    lower_intervals.append(i - (1.96 * st_dev))
    upper_intervals.append(i + (1.96 * st_dev))

plt.figure(figsize=(12,8))
plt.plot(sample_means, label='Mean')
plt.plot(upper_intervals, label='Upper Bound')
plt.plot(lower_intervals, label='Lower Bound')
plt.show()

pop_mean = np.mean(X)
boot_mean = np.mean(sample_means)

