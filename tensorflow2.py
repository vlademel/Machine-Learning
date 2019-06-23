# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 13:32:23 2019

@author: Vlad
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator

def error_rate(p, t):
    return np.mean(p != t)

def main():
    max_iter = 20
    print_period = 10
    
    lr = 0.00004
    reg = 0.01

    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    
    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz
    
    M1 = 300
    M2 = 200
    K = 10
    W1_init = np.random.randn(D, M1)/ 28
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)
    
    #Create tensorflow variables
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    
    Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
    Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
    Yish = tf.matmul(Z2, W3) + b3
    
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))
    
    #RMS prop optimiser includes momentum
    train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)
    
    predict_op = tf.argmax(Yish, 1)
    
    LL = []
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        
        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
                
                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j & print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    LL.append(test_cost)
                    
    plt.plot(LL)
    plt.show()
                    
if __name__ == '__main__':
    main()    