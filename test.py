#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:19:12 2017

Exploring the basics of Neural Networks

Simple base was established off of stanfords UFLDL Tutorial found here:
http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial 
As well found from the series by Welch labs

The rest is all me

Test Matrix:
    X = np.array(([3,5], [5,1], [10,2]), dtype=float)
    X = X/np.amax(X, axis=0) 
    y = np.array(([75], [82], [93]), dtype=float)
    y = y/100

To DO: 
        Update documentation
        

@author: christopherstewart
"""
import numpy as np

class Neural_Network(object):
    """
     General Steps in Network
     ---------------------------------:
     For [3x1] Input -> [1x1] Output
     _______________________________________
    |                                       |
    |    (1) [3x1][1x3] = [3x3]             |
    |           Input(dot)W1 = Z2           |
    |    (2) [3x3][3x1] = [3x1]             |
    |           Z2(dot)W2 = Z3              |
    |    (3) [3x1]^T[3x1] = [1x1]           |
    |           a2^T(dot)Z3 = Output        |
    |_______________________________________|
     
    """
    def __init__(self):
        """
        Simple network used for learning purposes
        ____________________________________________________________
        
        Hyperparameters (not to be changed by the network)
            -> 3x1 Input matrix
            -> 1x1 Output matrix
            -> 3 hidden layers:
                    Weight 1 = W1
                    Weight 2 = W2
                    
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
        """
        Simple activation function
        ____________________________________________________________
        Input:  -> Network
                            (Current Initilized Neural Network)
                -> Z
                            (This is a sub J value)
                            
        Output:
                -> J(Z)
                            (Value at point Z in our activation function)
        """
        return 1/(1+np.exp(-z))
    
    def sigmoidPrime(self,z):
        """
        Gradient of our simple activation function
        ____________________________________________________________
        Input:  -> Network
                            (Current Initilized Neural Network)
                -> Z
                            (This is a sub J value)
                            
        Output:
                -> dJ/dZ
                            (Value at point Z in our activation function)
        
        """
        return np.exp(-z)/((1+np.exp(-z))**2)
        
    def costFunction(self,X,y):
        """
        Finds the cost where cost is how wrong your current fit is.
        
        ____________________________________________________________
        Input:  -> Network
                            (Current Initilized Neural Network)
                 -> X
                            (Left side Input matrix)
                 -> y
                            (Right side Input matrix)
                
        Output: -> J
                            (Cost/How good of a fit the weights provide)
        """
        self.yHat = self.forward(X)
        J=0.5*sum((y-self.yHat)**2)
        
        return J
    
    def costFunctionPrime(self, X, y):
        """
        Finds the cost where cost is how wrong your current fit is.
        
        ____________________________________________________________
        Input:  -> Network
                            (Current Initilized Neural Network)
                 -> X
                            (Left side Input matrix)
                 -> y
                            (Right side Input matrix)
                
        Output: -> dJdW1
                            (Derivative of J with respect to W1 [2x3])
                -> dJdW2
                            (Derivative of J with respect to W2 [3x1])
        """
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat),self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T,delta2)
        
        return dJdW1, dJdW2
    
    def getParams(self):
        """
        Sets the paramters to a simple row vector composed of W1 and W2
        ____________________________________________________________
        
        Input:  -> Network
                            (Current Initilized Neural Network)
        Output: -> Parameters
        """
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        
        """
        Set W1 and W2 using single paramater vector
        ____________________________________________________________
        Input:  -> Network
                            (Current Initilized Neural Network)
                -> params
                            (Usually weights in the form of a single row matrix)
        Output: -> None
                            (Simply updates the values of W1 and W2 of the 
                             network)
        __________Steps__________
        (1) W1_end ->  6     by: 3*2
        (2) W1     ->  [2x3] by: np.shape([0:6],(2,3)) 
        (3) W2_end ->  9     by: 6 + 3*1
        (4) W2     ->  [3x1] by: npshape(params[6:9],(3,1))
        _________________________
        """
        
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, 
                             self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, 
                             self.outputLayerSize))
        
    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumGrad(Network, X, y):
        """
        Test class used to insure that the we are on the right track.
        
        ____________________________________________________________
        
        Input:  -> Network
                            (Current Initilized Neural Network)
                -> X
                            (Input matrix)
                -> y
                            (Right side Input matrix))
        Output: -> numgrad
                            (This vector that is returned should be very close
                             to what Neural.computeGradient(X,y) returns)
        """
        paramsInitial = Network.getParams()
        #creates 2 empty vectors
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        #small distance e
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
def checkFit(grad,numGrad):
    from numpy import linalg as LA
    """
    Taken from Stanford UFLDL Tutorial:
    ____________________________________________________________
        
        Input:  -> grad
                        (Gradient found through the neural network)
                -> numGrad
                        (Gradient calculated by the function computeNumGrad)
        Output:
                -> Fit
                        (How close the Neural Network is to NumGrad)
    
    """
    return LA.norm(grad-numGrad)/LA.norm(grad+numGrad)