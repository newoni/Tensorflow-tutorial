# <20.3.16> by KH
'''
gradient descent 연습


'''
import numpy as np

X = np.array([1,2,3])
Y = np.array([1,2,3])

def cost_func(W,X,Y):
    c = 0
    for i in range(len(X)):
        c += ((W*X[i])-Y[i])**2

    return c/len(X)

for feed_W in np.linspace(-3,5, num= 15):
    curr_cost = cost_func(feed_W, X, Y)
    print("{:6.3}|{:10.5}".format(feed_W, curr_cost))


###################################################
import tensorflow as tf

tf.random.set_seed(0)

x_data = np.array([1.,2.,3.,4.])
y_data = np.array([1.,3.,5.,7.])

W = tf.Variable(tf.random.normal([1],-100,100))

for step in range(300):
    hypothesis = W*x_data
    cost = tf.reduce_mean(tf.square(hypothesis-y_data))

    alpha = 0.01
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W,x_data) -y_data, x_data))
    descent = W - tf.multiply(alpha,gradient)
    W.assign(descent)

    if step%10 == 0:
        print("{:5}|{:10.6f}|{:10.6f}".format(
            step, cost.numpy(),W.numpy()[0]))