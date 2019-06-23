# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:44:08 2019

@author: Vlad
"""

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator

class ANN:
    def __init__(self, Xtrain, Ytrain, Xtest, Ytest):
        print("Initialising NN...")
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        
        self.Ytest_ind = y2indicator(self.Ytest)
        self.Ytrain_ind = y2indicator(self.Ytrain)

    def error_rate(self, p, t):
        return np.mean(p != t)
    
    def relu(self, a):
        return a * (a > 0)
    
    def model(self, lr, reg, M):
        max_iter = 20
        print_period = 10
        N, D = self.Xtrain.shape
        batch_sz = 500
        n_batches = N // batch_sz
        
        K = 10
        W1_init = np.random.randn(D, M)/ 28
        b1_init = np.zeros(M)
        W2_init = np.random.randn(M, K) / np.sqrt(M)
        b2_init = np.zeros(K)
        
        #Create theano variables
        thX = T.matrix('X')
        thT = T.matrix('T')
        W1 = theano.shared(W1_init, 'W1')
        b1 = theano.shared(b1_init, 'b1')
        W2 = theano.shared(W2_init, 'W2')
        b2 = theano.shared(b2_init, 'b2')
    
        thZ = self.relu( thX.dot(W1) + b1 )
        thY = T.nnet.softmax( thZ.dot(W2) + b2 )
        
        cost = -(thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() +(W2*W2).sum() + (b2*b2).sum())
        prediction = T.argmax(thY, axis=1)
        
        update_W1 = W1 - lr*T.grad(cost, W1)
        update_b1 = b1 - lr*T.grad(cost, b1)
        update_W2 = W2 - lr*T.grad(cost, W2)
        update_b2 = b2 - lr*T.grad(cost, b2)
        
        train = theano.function(
                inputs=[thX, thT],
                updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)],
                )
        
        get_prediction = theano.function(
                inputs=[thX, thT],
                outputs=[cost, prediction],
                )
        
        LL = []
        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = self.Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = self.Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
                
                train(Xbatch, Ybatch)
                if j % print_period == 0:
                    cost_val, prediction_val = get_prediction(self.Xtest, self.Ytest_ind)
                    err = self.error_rate(prediction_val, self.Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, err))
                    LL.append(cost_val)
                    
#        plt.plot(LL)
#        plt.show()
        return lr, reg, M, LL[-1]
    
    def grid_search(self, M, lrs, regs):
        learning_rates = []
        reg_values = []
        node_amounts = []
        LLs = []
        
        for m in M:
            for lr in lrs:
                for reg in regs:
                    learn, regularise, nodes, LL = self.model(lr, reg, m)
                    learning_rates.append(learn)
                    reg_values.append(regularise)
                    node_amounts.append(nodes)
                    LLs.append(LL)
        
        index = LLs.index(min(LLs))
        return learning_rates[index], reg_values[index], node_amounts[index], LLs[index]    

def main():       
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
#    Ytrain_ind = y2indicator(Ytrain)
#    Ytest_ind = y2indicator(Ytest)
    
    #Find optimal hyperparams    
    M = [100, 200, 300]
    lrs = [0.00001, 0.000001, 0.0001]
    regs = [0.1, 0.01, 0.001]
    
    ann = ANN(Xtrain, Ytrain, Xtest, Ytest)
    
    lr, reg, M, LL = ann.grid_search(M, lrs, regs)
    print("Found optimal values: lr={}, reg={}, M={} at a cost of {}".format(lr, reg, M, LL))
    
    
    
if __name__ == '__main__':
    main()
    