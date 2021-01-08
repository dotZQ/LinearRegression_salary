#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 17:03:18 2021

@author: zq314159
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataframe = pd.read_csv("/Users/zq314159/Desktop/ML/dataset/regression/salary.csv")
train_set,test_set = train_test_split(dataframe,test_size=0.6,random_state=42)
train_set.plot.scatter(x='YearsExperience',y='Salary')

x = train_set['YearsExperience'].to_frame()
y = train_set['Salary'].to_frame()

x_test = test_set['YearsExperience'].to_frame()
y_test = test_set['Salary'].to_frame()

model = LinearRegression()
model.fit(x,y)

score = model.score(x_test,y_test)

xSample = np.arange(0,12).reshape(-1,1)
ySample = model.predict(xSample)

test_set.plot.scatter(x='YearsExperience',y='Salary')
plt.plot(xSample,ySample,color='red',linewidth=3)

