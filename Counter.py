# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:21:46 2016

@author: naveenkumar2703
Code that learns to count 3 digit binary number using neural networks
"""
import numpy as np
from NeuralNetwork import *

# 0.1 indicates 0 and 0.9 indicates 1. the values are transformed for efficient learning
inputdata = np.array([np.array([0.1,0.1,0.1]),np.array([0.1,0.1,0.9]),np.array([0.1,0.9,0.1]),np.array([0.1,0.9,0.9]),np.array([0.9,0.1,0.1]),np.array([0.9,0.1,.9]),np.array([.9,.9,0.1]),np.array([.9,.9,.9])])
outputdata = np.array([np.array([0.1,0.1,0.9]),np.array([0.1,0.9,0.1]),np.array([0.1,0.9,0.9]),np.array([0.9,0.1,0.1]),np.array([0.9,0.1,.9]),np.array([.9,.9,0.1]),np.array([.9,.9,.9]),np.array([0.1,0.1,0.1])])


trainigdatas = []
for ind in range(len(inputdata)):
    training_data = [inputdata[ind],outputdata[ind]]
    trainigdatas.append(training_data)

net = NeuralNetwork(3, [3, 3, 3])


i = 0

while i < 3000:
    net.train(trainigdatas, 1, 0.1, 10, 1)
    
    if i%1000 == 0:
        print('Completed iterations - ' + str(i))
        for item,oitem in zip(inputdata,outputdata):
            out = net.predict(item)
            print('ex- '+ str(oitem) + 'act out -' + str(out))
    i += 10
        



