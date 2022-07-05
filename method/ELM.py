import numpy as np
import pandas as pd
class ELM(object):

    def __init__(self, inputSize, outputSize, hiddenSize, activation, x_train, x_test, y_train, y_test):
        self.inputSize = inputSize  # no of features
        self.outputSize = outputSize  # no of classes
        self.hiddenSize = hiddenSize  # no of hidden neurons
        self.activation = activation  # activation function
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.Wout = 0  # output weights

        # self.X = 0 #
        # self.inputSize = x_train.shape[1]

        self.Win = np.random.normal(size=[inputSize, hiddenSize])  # init weights
        #biases initialization
        self.biases = np.random.normal(self.hiddenSize)

    def initWeights(self):
        self.Win = np.random.normal(size=[self.inputSize, self.hiddenSize])  # init weights
        self.biases = np.random.normal(self.hiddenSize)

    def setWeights(self, Win):
        #we discard last hiddenSize elements because those are biases
        self.Win = np.array(Win[0:self.inputSize*self.hiddenSize]).reshape(self.inputSize,self.hiddenSize)
        self.biases = np.array(Win[self.inputSize*self.hiddenSize:])

        #self.Win = np.array(Win).reshape(self.inputSize,self.hiddenSize)

     #implementation with biases
     #ovo proveriti

    def input_to_hidden(self, x):
        a = np.dot(x, self.Win)

        for j in range(len(a)):
            for i in range(self.hiddenSize):
                a[j,i] = a[j,i] + self.biases[i]



        if self.activation == 'relu':
            a = np.maximum(a, 0, a)  # ReLU  #3rd arugment a specifies that result is stored in a
        elif self.activation == 'sigmoid':
            a = 1 / (1 + np.exp(-a))
        else:
            a = np.maximum(a, 0, a)
        return a

    #implementation without biases
    def input_to_hidden1(self, x):
        a = np.dot(x, self.Win)
        if self.activation == 'relu':
            a = np.maximum(a, 0, a)  # ReLU  #3rd arugment a specifies that result is stored in a
        elif self.activation == 'sigmoid':
            a = 1 / (1 + np.exp(-a))
        else:
            a = np.maximum(a, 0, a)
        return a

    def train(self):
        X = self.input_to_hidden(self.x_train)
        Xt = np.transpose(X)
        self.Wout = np.dot(np.linalg.pinv(np.dot(Xt, X)), np.dot(Xt, self.y_train))
       # print('Output weights shape: {shape}'.format(shape=self.Wout.shape))

    def predict(self, x):
        x = self.input_to_hidden(x)
        y = np.dot(x, self.Wout)
        return y

    def predict1(self):  # predict metoda koja vec koristi test
        # returns tuple (predict_proba per class, and predict as integers)
        x = self.input_to_hidden(self.x_test)
        y_proba = np.dot(x, self.Wout)
        correct = 0
        total = y_proba.shape[0]
        for i in range(total):
            predicted = np.argmax(y_proba[i])
            test = np.argmax(self.y_test[i])
            correct = correct + (1 if predicted == test else 0)
        #print('Accuracy: {:f}'.format(correct / total))
        y = np.zeros((len(y_proba), y_proba.shape[1]))
        for i in range(len(y_proba)):
            y[i][np.argmax(y_proba[i])] = 1
        #classification error
        error = 1-(correct/total)

        return (error,y_proba,y)
