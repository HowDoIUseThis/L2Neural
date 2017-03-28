#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:19:12 2017

Exploring the basics of Neural Networks

Simple base was established off of stanfords UFLDL Tutorial found here:
http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial 
As well found from the series by Welch labs

The rest is all me

@author: christopherstewart
"""
import numpy as np

class Neural_Network(object):
    def __init__(self):
        """
        Simple network used for learning purposes
        
        Hyperparameters (not to be changed by the network)
            -> 3x1 Input matrix
            -> 1x1 Output matrix
            -> 3 hidden layers:
                    Weight 1 = W1
                    Weight 2 = W2
                    
        ________Steps__________
        (1) [3x1][1x3] = [3x3]
            Input(dot)W1 = Z2
        (2) [3x3][3x1] = [3x1]
            Z2(dot)W2 = Z3
        (3) [3x1]^T[3x1] = [1x1]
            a2^T(dot)Z3 = Output
        
        """
        self.inputLayerSize = 2
        self.outputLayerSize =1
        self.hiddenLayerSize =3
        #Weights 
        self.W1 = np.random.randn(self.inputLayerSize,
                                  self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,
                                  self.outputLayerSize)
        
    def forward(self, X):
        """
        Moves a step forward in Network
        
        Inputs: -> self
                        (Current Initilized Neural Network)
                -> X
                        (Data Points)
        """
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
        
        
    def sigmoid(self, z):
        #for this test we will use signmoid as our activation
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        #Gradient of sigmiod
        return np.exp(-z)/((1+np.exp(-z))**2)
        
    def costFucntionPrime(self, X, y):
        #Calculates derivative with respect to W1 and W2
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T,delta2)
        
        return dJdW1, dJdW2
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(Network, X, y):
        """
        Test class used to insure that the we are on the right track
        
        Input:  -> Network
                -> X
        """
        paramsInitial = Network.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            Network.setParams(paramsInitial + perturb)
            loss2 = Network.costFunction(X, y)
            
            Network.setParams(paramsInitial - perturb)
            loss1 = Network.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        Network.setParams(paramsInitial)

        return numgrad 