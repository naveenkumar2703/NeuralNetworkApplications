# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 00:00:44 2016
Core file for neural networks which has functions to train and predict.
@author: naveenkumar2703
"""
import numpy as np
from collections import deque

class NeuralNetwork(object):
    def __init__(self, numOfLayers, nodesInLayers):
        self.numOfLayers = numOfLayers
        self.networkStructure = nodesInLayers
        self.bias = [np.random.randn(y, 1) for y in nodesInLayers[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(nodesInLayers[:-1], nodesInLayers[1:])]
        self.prevdelweight = np.zeros_like(self.weights) #[np.random.randn(y, x) for x, y in zip(nodesInLayers[:-1], nodesInLayers[1:])]
        self.prevdelbias = np.zeros_like(self.bias) #[np.random.randn(y, 1) for y in nodesInLayers[1:]]     
                
    def predict(self, inputvec):
        return self.feedforward(inputvec)[0]#appending 1 for bias and returning output at zero index

    # trains a neural network based on given training examples, learning rate and batch sizes.
    def train(self, training_examples, learningRate, momentum, training_iterations, batchsize):
        for index in range(training_iterations):
            layer_error_batch = deque()
            for exampleindex in range(len(training_examples)):
                inputvec = training_examples[exampleindex][0]
                expectedout = training_examples[exampleindex][1]
                predictedout, layeroutputs = self.feedforward(inputvec)
                layer_errors = self.backpropogate(inputvec, expectedout, predictedout, layeroutputs)
                if len(layer_error_batch) == 0:
                    layer_error_batch.extend(layer_errors)
                else:
                    for errorIndex in range(len(layer_error_batch)):
                        layer_error_batch[errorIndex] = layer_error_batch[errorIndex] + layer_errors[errorIndex]
                    
                if (exampleindex + 1) % batchsize == 0:
                    self.updateLayerWeights(learningRate, layer_errors, layeroutputs, momentum, batchsize)
                    layer_error_batch = deque()#reinitializing for next batch
        
    def feedforward(self, inputval):
        output = None
        inputvec = np.array(inputval)
        layeroutputs = [inputvec]
        for weight, bias in zip(self.weights, self.bias):
            output =  self.computeSigmoid(inputvec, weight, bias)
            layeroutputs.append(output)
            inputvec = output#np.append(output, 1)#adding 1 for next layer bias at end
        return output,layeroutputs
    
    def backpropogate(self, inputvec, expectedOut, predictedOut, layeroutputs):
        layer_errors = deque() 
        network_error = self.computeNetworkError(expectedOut, predictedOut)
        layer_errors.appendleft(network_error) 
        prevLayerError = network_error
        for index in range(len(self.weights)):
            if index < (len(self.weights) - 1):
                currLayerOut = layeroutputs[-(index + 2)]#accessing hidden layer output
                layerWeight = self.weights[-(index + 1)]#accessing hidden layer weight to next layer
                prevLayerError = self.computeHiddenLayerError(currLayerOut, layerWeight, prevLayerError)
                layer_errors.appendleft(prevLayerError)
        return layer_errors
    
    def computeNetworkError(self, expectedOut, predictedOut):
        unit_vec = [1] * len(predictedOut)
        return predictedOut * (unit_vec - predictedOut) * (expectedOut - predictedOut)
    
    def computeHiddenLayerError(self, hidLayerOut, hidLayerWeight, prevLayerError):
        unitVec = np.array([1] * len(hidLayerOut))
        hidOutDifProd = hidLayerOut * (unitVec - hidLayerOut)#compute osubh(1-osubh)
        sumOfWeightAndBackPropError = np.dot(hidLayerWeight.transpose(),prevLayerError) #compute sum of wsubkh * delta sub kh
        hiddenLayerError = sumOfWeightAndBackPropError * hidOutDifProd
        return hiddenLayerError
        
    def updateLayerWeights(self, learningRate, layerErrors, layeroutputs, momentum, batchsize):
        for index in range(len(layerErrors)):
            lerror = layerErrors[index]
            inputForNextLayer = layeroutputs[index]
            delWeight = (learningRate/batchsize) * np.dot(np.reshape(lerror, (len(lerror),1)), np.reshape(inputForNextLayer, (len(inputForNextLayer),1)).transpose())
            berror = (learningRate/batchsize) * lerror#bias error is same as del error term
            delWeight = delWeight + (momentum * self.prevdelweight[index])
            berror = berror +(momentum * self.prevdelbias[index])
            self.weights[index] = self.weights[index] + delWeight
            self.bias[index]    = self.bias[index] + berror
            self.prevdelbias[index] = berror
            self.prevdelweight[index] = delWeight

    def computeSigmoid(self, in_vec, weight, bias):
        dotProd = np.dot(weight, in_vec)
        netval = []
        for index in range(len(dotProd)):
            netval.append((dotProd[index] + bias[index][0]))#elementwise array addition not working
        return 1/(1 + np.exp(-(np.array(netval))))