# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:52:10 2019

@author: Vlad
"""
import numpy as np
import pandas as pd
import os
import sys

#def initial_processing():
#    if not os.path.exists(r'C:\Users\Vlad\Documents\Python Scripts\Kaggle\Facial Expressions\fer2013.csv'):
#        print("File 'fer2013.csv' does not exist")
#        sys.exit()
#    
#    
#    file = r'C:\Users\Vlad\Documents\Python Scripts\Kaggle\Facial Expressions\fer2013.csv'
#    data = np.genfromtxt(file, delimiter=',', dtype=None)
#    np.random.shuffle(data)
#    
#    rows, cols = data.shape
#    Y = np.zeros((rows, 1))
#    tst_or_trn = np.zeros((rows, 1))
#    for i in range(rows):
#        try:
#            tst_or_trn[i] = data[i][2]
#            Y[i] = data[i][0].astype(np.float32)
#        except:
#            continue
#    
#    num_cols = 2304 # splitting up into list gives this number
#    X = np.zeros((rows, num_cols))
#    for i in range(rows):
#        try:
#             X[i] = data[i][1].astype(str).split(' ')
#             print("Parsing {}".format(i))
#             X = X.astype(np.float32)
#        except:
#            continue
#        
#    np.savetxt('Xvalues.csv', X, delimiter=',')
#    np.savetxt('Yvalues.csv', Y, delimiter=',')
#    np.savetxt('Type.csv', tst_or_trn, delimiter=',')
    
def getData(balance_ones=True):
    if not os.path.exists(r'C:\Users\Vlad\Documents\Python Scripts\Kaggle\Facial Expressions\fer2013.csv'):
        print("File 'fer2013.csv' does not exist")
        sys.exit()
    
    
    file = r'C:\Users\Vlad\Documents\Python Scripts\Kaggle\Facial Expressions\fer2013.csv'
    X = []
    Y = []
    first = True
    for line in open(file):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
    
    X, Y = np.array(X) / 255.0, np.array(Y)
    
    if balance_ones:
        #Balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))
    return X, Y


#def get_normalised_data(train_split = 0.7):
#    Yfile = r'C:\Users\Vlad\Documents\Python Scripts\Kaggle\Facial Expressions\Yvalues.csv'
#    Xfile = r'C:\Users\Vlad\Documents\Python Scripts\Kaggle\Facial Expressions\Xvalues.csv'
#    
#    X = np.genfromtxt(Xfile, delimiter=',', dtype=None)
#    Y = np.genfromtxt(Yfile, delimiter=',', dtype=None)
#    
#    #Split across train/test and then normalise data
#    train_length = np.round(len(X)*train_split)
#    test_length = len(X) - train_length
#    
#    Xtrain = X[:train_length]
#    Xtest = X[train_length:]
#    Ytrain = Y[:train_length]
#    Ytest = X[train_length:]
#    
#    mu = Xtrain.mean(axis=0)
#    std = Xtrain.std(axis=0)
#    np.place(std, std==0, 1)
#    Xtrain = (Xtrain - mu) / std
#    Xtest = (Xtest - mu) / std
#    
#    return Xtrain, Xtest, Ytrain, Ytest

def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

def getBinaryData():
    Y = []
    X = []
    first = True
    for line in open(file):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)
    

def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    ind = np.zeros((N, 7))
    for i in range(N):
        ind[i, Y[i]] = 1
    return ind
    
def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)

def cost(T, Y):
    return -(T*np.log(Y)).sum()

def cost2(T, Y):
    N = len(T)
    return -(T*np.log(Y[np.arange(N)]), T).sum()

def error_rate(targets, predictions):
    return np.mean(targets != predictions)
    


