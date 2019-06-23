# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:30:30 2019

@author: Vlad
"""

"""Building a logistic regression in Tensorflow"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator

def error_rate(p, t):
    return np.mean(p != t)

def main():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)
    
    lr = 0.00004
    reg = 0.01
    N, D = Xtrain.shape
    K = 10
    
    max_iter = 1000
    batch_sz = 500
    n_batches = N // batch_sz
    print_period = 10
    
    W_init = np.random.randn(D, K)
    b_init = np.random.randn(K)
    
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W = tf.Variable(W_init.astype(np.float32))
    b = tf.Variable(b_init.astype(np.float32))
    
    Yish = tf.matmul(X, W) + b
    
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))
    
    #train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
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
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    LL.append(test_cost)
                    
    plt.plot(LL)
    plt.show()
        
    

if __name__ == '__main__':
    main()