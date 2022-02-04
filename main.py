import math

import numpy as np
from math import exp
import pandas as pd
import matplotlib.pyplot as plt
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

def print_hi(name):
    data = pd.read_csv('/Users/danielemorganti/Desktop/uni/terzo_anno/1s_AI/BackPropagation/archive/mnist_train.csv')
    data = np.array(data)
    dataset = discardExcept(data, 1, 9)
    NN = ThreeLayerNeuralNetwork(784, 300, 1)
    p = 0
    N_epochs = 5
    andamentoloss = []
    loss = []
    for j in range(N_epochs):                                   #
        np.random.shuffle(dataset)                              #
        for i in range(10000):                                  #
                                                                #
            X = dataset[i, 1:785]/255                           #
            Y = (dataset[i, 0] == 1)                            #traning
                                                                #
            forwardPropagation(NN, X)                           #
            backPropagation(NN, Y, 0.001)                       #
            loss.append(calcloss(Y, NN.A2))                     #
                                                                #
            if i%100==0:                                        #
                print(sum(loss)/len(loss))                      #
                andamentoloss.append(sum(loss)/len(loss))       #
                print("------")                                 #
                                                                #
    np.random.shuffle(dataset)                          #
    for i in range(1000):                               #
        X = dataset[i, 1:785]/255                       #
        Y = (dataset[i, 0] == 1)                        #
                                                        #testing
        forwardPropagation(NN, X)                       #
                                                        #
        if NN.A2 >= 0.5 and Y == 1:                     #
            p += 1                                      #
        if NN.A2 < 0.5 and Y == 0:                      #
            p += 1                                      #

    plt.plot(range(N_epochs * 100),andamentoloss)
    plt.xlabel("iterations x100")
    plt.ylabel("loss")
    plt.show()
    print("percentage of good prediction in the last 1000 examples")
    print(p/10)

class ThreeLayerNeuralNetwork:
    def __init__(self, dl0, dl1, dl2):
        self.A0 = np.zeros(dl0)
        self.W1 = np.random.uniform(-1/np.sqrt(dl0+dl1),1/np.sqrt(dl0+dl1),(dl1, dl0))
        self.A1 = np.zeros(dl1)
        self.W2 = np.random.uniform(-1/np.sqrt(dl1+dl2),1/np.sqrt(dl1+dl2),(dl2, dl1))
        self.A2 = np.zeros(dl2)
        self.dl0 = dl0
        self.dl1 = dl1
        self.dl2 = dl2
        self.Z1 = np.ones(dl1)
        self.Z2 = np.ones(dl2)

def forwardPropagation(NN, X):

    NN.A0 = X
    NN.Z1 = NN.W1.dot(X)
    NN.A1 = ReLU(NN.Z1)
    NN.Z2 = NN.W2.dot(NN.A1)
    NN.A2 = sigmoid(NN.Z2)

def backPropagation(NN,Y,lr):

    dZ2 = NN.A2 - Y
    dW2 = dZ2 * NN.A1
    dZ1 = np.dot(NN.W2.T, dZ2) * dReLU(NN.Z1)
    dW1 = np.dot(dZ1.reshape(NN.dl1,1), NN.A0.reshape(1,NN.dl0))

    NN.W1 = NN.W1 - lr * dW1
    NN.W2 = NN.W2 - lr * dW2

def ReLU(x):
    return np.maximum(0,x)

def dReLU(x):
    return x>0

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dSigmoid(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

def discardExcept(dataset, a ,b):
    j = 0
    for i in range(dataset.shape[0]):
        if dataset[i,0] == a or dataset[i,0] == b:
            j = j+1
    newDataset = np.zeros((j,785))
    j=0
    for i in range(dataset.shape[0]-1):
        if dataset[i,0] == a or dataset[i,0] == b:
            newDataset[j] = dataset[i]
            j = j+1
    return newDataset

def calcloss(Y, A):
    return -((1-Y)*np.log(1-A)+Y*np.log(A))

if __name__ == '__main__':
    print_hi('PyCharm')

