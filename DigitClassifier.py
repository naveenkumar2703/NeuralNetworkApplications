# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:29:43 2016
This file trains a neural network to classify digits based on images and stores the weights of trained network as pkl.
@author: naveenkumar2703
"""

import os
from NeuralNetwork import *
import datetime
import numpy as np
import random
import copy
from random import randint
import pickle
import math

print('started execution')
a = datetime.datetime.now()
print(a)
def constructOutput(number):
    #output = np.zeros(10)
    #output[int(number)] = 1.0
    output = []
    for index in range(10):
        if index == int(number):
            output.append(0.9)
        else:
            output.append(0.1)
    return np.array(output)

def validateOutput(predicted, expected, pr):
    #validationSucess = True
    #print(predicted)
    #print(expected)
    predictedMax = predicted[0]
    expectedMax = expected[0]
    predictedMaxindex = 0
    expectedMaxIndex = 0
    for index in range(10):
        if predicted[index] > predictedMax:
            predictedMax = predicted[index]
            predictedMaxindex = index
        if expected[index] > expectedMax:
            expectedMaxIndex = index
            expectedMax = expected[index]
            #print(predicted)
            #print(expected)            
    validationSucess = (predictedMaxindex == expectedMaxIndex)
    if pr:
        if not (validationSucess):
            'a'            
            print(expectedMaxIndex)
            print(predictedMaxindex)
            print(predicted)
            print(expected)
        """else:
            print('success')
            print(predictedMaxindex)
            print(predicted)"""
    return validationSucess

net = NeuralNetwork(3, [784, 800, 10])
trainingOutputs = []
trainingInputs = []
training_datas = []
with open(str(os.getcwd()+'/sudo.csv'), 'r', encoding='utf-8') as input_file:
    numberOfData = 0 
    
    for line in input_file:
        numberOfData += 1
        if line[0].isnumeric():
            inputString = line[2:]
            trainingInputFormatted = inputString.split(',')
            trainingInput = []
            for item in trainingInputFormatted:
                #trainingInput.append(int(item)/255)
                trainingInput.append((math.ceil(int(item)/32))/8)
            trainingInputs.append(np.array(trainingInput))
            trainingOutput = constructOutput(line[0])
            trainingOutputs.append(trainingOutput)
            training_data = [np.array(trainingInput),trainingOutput]
            training_datas.append(training_data)
            
learning_rate = .3
batchsize = 10
training_iterations = 0
training_iter_inc = 10
inputForTrain = training_datas[:1800]
#print(len(inputForTrain))
inputForvalidation = trainingInputs[1800:]
outputForvalidation = trainingOutputs[1800:]
#inputForvalidation.extend(trainingInputs[35000:])
#outputForvalidation.extend(trainingOutputs[35000:])
        


accuracy = 0
prevaccuracy = 0
checkpointcount = 0
highestaccuracy = 0
accuratenet = net
momentum = 0.1

def writeOutputs():
    totSucceessCount = 0
    for itemindex in range(len(trainingInputs)):
        if validateOutput(accuratenet.predict(trainingInputs[itemindex]), trainingOutputs[itemindex], highestaccuracy > 0.9925):
            totSucceessCount += 1
    print('Accuracy on all train data is:'+str(totSucceessCount/len(trainingInputs)))

    targetInputs = []    
    with open(str(os.getcwd()+'/test.csv'), 'r', encoding='utf-8') as testfile:
        numberOfData = 0 
    
        for line in testfile:
            numberOfData += 1
            if line[0].isnumeric():
                trainingInputFormatted = line.split(',')
                testInput = []
                for item in trainingInputFormatted:
                    #testInput.append(int(item)/255)
                    testInput.append((math.ceil(int(item)/32))/8)
                targetInputs.append(np.array(testInput))
            
    with open(str(os.getcwd()+'/'+str(datetime.datetime.now())+'output_'+str(highestaccuracy)+'.csv'), 'w', encoding='utf-8') as outfile:
        outfile.write('ImageId,Label\n')
        for inindex in range(len(targetInputs)):
            predicted = accuratenet.predict(targetInputs[inindex])
            predictedMax = predicted[0]
            predictedMaxindex = 0
            for index in range(10):
                if predicted[index] > predictedMax:
                    predictedMax = predicted[index]
                    predictedMaxindex = index
            outfile.write(str(inindex+1)+','+str(predictedMaxindex)+'\n')
    print('done writing to file with highest accuray - '+str(highestaccuracy))
    with open(str("first.pkl"), "wb") as f:
        pickle.dump(accuratenet,f,pickle.HIGHEST_PROTOCOL)
    
learningRateChanged = False
changeNow = False
while accuracy < 0.99997 and training_iterations < 10000:
    succeessCount = 0
    prevaccuracy = accuracy
    random.shuffle(inputForTrain)
    net.train(inputForTrain, learning_rate, momentum, training_iter_inc, batchsize)
    training_iterations = training_iterations + training_iter_inc
    for itemindex in range(len(inputForvalidation)):
        if validateOutput(net.predict(inputForvalidation[itemindex]), outputForvalidation[itemindex], False):
            succeessCount += 1
       #print(itemindex)
       
    print(succeessCount)
    print(len(inputForvalidation) )
    accuracy = (succeessCount)/len(inputForvalidation)
    if prevaccuracy == 0:
        prevaccuracy = accuracy
        checkpointcount = int(succeessCount)
        highestaccuracy = accuracy
        accuratenet = copy.deepcopy(net)
    if highestaccuracy < accuracy:
        accuratenet = copy.deepcopy(net)
        highestaccuracy = accuracy
    print('##################################'+str(datetime.datetime.now())+'#################################')
    print('Accuracy at end of iterations '+str(training_iterations) + 'is:' + str(accuracy))    
    print('eta:' + str(learning_rate))
        
    if(training_iterations % 500 == 0 and learning_rate > 0.01 and not learningRateChanged):
        print('changing eta')
        learning_rate = learning_rate/1.05
        learningRateChanged = False
            
    elif training_iterations % 500 == 0:
        learningRateChanged = False

    if (highestaccuracy - accuracy) > 0.75 and (prevaccuracy - accuracy) >= 0 and learning_rate > 0.05:
        if changeNow or (highestaccuracy - accuracy) > 0.4:
            print('slowing down as highest is :'+str(highestaccuracy))
            learning_rate = learning_rate/1.25
            learningRateChanged = True
            changeNow = False
        else:
            changeNow = True
    else:
        changeNow = False
        
    if learning_rate < 0.05:
        learning_rate = randint(2,6)
    
    if training_iterations % 250 == 0:
        writeOutputs()
           

print('Ended when accuracy dropped from '+str(prevaccuracy)+' to '+str(accuracy))
print(learning_rate)
print('no of iterations' + str(training_iterations))
print('highest accuracy - ' + str(highestaccuracy)+'@iter'+str(training_iterations))
writeOutputs()
b = datetime.datetime.now()
print(a)
print(b)


#writeOutputs()
print('All done')
        