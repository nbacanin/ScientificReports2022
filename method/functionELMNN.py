
#ELM funkcija sa optimizacijom broja neurona


import math
import numpy as np
from utilities.load_dataset import load_dataset
from ml_models.ELM import  ELM

class ELMFunction:
    def __init__(self, D, no_classes,x_train,x_test,y_train,y_test, lb_w=-1,ub_w=1,lb_nn=1,ub_nn=1000,nn=50):
        #D is number of parameters (input feature size * number of hidden unit neurons)
        #lb_w, ub_w - lower and upper bound of weights
        #lb_nn, ub_nn lower and upper bounds of neurons in the hidden layer
        #no_classes number of classes in the dataset
        self.no_classes = no_classes
        self.nn = nn

        #D treba postaviti na max broj D iz spoljasnjeg koda, sto je upper bound za broj neurona*feature_size + upper_bound za broj neurona


        #self.x_train, self.y_train, self.x_test, self.y_test = load_dataset(path_train, path_test, self.no_classes, normalize, test_size)

        self.x_train=x_train
        self.y_train=y_train
        self.x_test=x_test
        self.y_test = y_test

        #ovo nam koristi za solution
        self.y_test_length = len(y_test)

        self.D = D
        #D is weighs*input features length + 1
        #in second experiment we use NN as well as the first argument of solution
        self.lb_w = lb_w
        self.ub_w = ub_w
        self.lb_nn= lb_nn
        self.ub_nn = ub_nn

        self.ub = [None] * self.D
        self.lb = [None] * self.D
        self.minimum = 0
        self.solution = '(0,...,0)'
        self.name = "ELMFunction"

        #ub[0],lb[0] is for hidden neurons size

        self.lb[0] = lb_nn
        self.ub[0] = ub_nn


        for i in range(1, self.D):
            self.lb[i] = self.lb_w
            self.ub[i] = self.ub_w
        #self.lb[0] = self.lb_nn
        #self.ub[0] = self.ub_nn





        #elm.initWeights()
        #elm.train()
        #y_proba, y = elm.predict1()



    def function(self, x):
        x = np.array(x)
        # initialization of ELM
        #treci argument je broj neurona, a broj neurona bice prvi parameter od x x[0], pa se uradi round
        #zaokruzujemo na integer
        nn = int(np.rint(x[0]))
        elm = ELM(self.x_train.shape[1], self.no_classes, nn, 'relu', self.x_train, self.x_test, self.y_train, self.y_test)

        #prvi argument je broj neurona, pa onda izbacujemo prvi argument iz niza

        elm.setWeights(x[1:])
        elm.train()
        #returns classification error, y_probabilities and y as dummy (1,0,0)
        error,y_proba,y = elm.predict1()
        return (error,y_proba,y)




