import numpy as np
import tensorflow as tf

# data and label
x1 = [ 73., 93., 89., 96., 73 ]
x2 = [ 80., 88., 91., 98., 66 ]
x3 = [ 75., 93., 90., 100., 70. ]
Y = [ 152., 185., 180., 196., 142. ]

w1 = tf.Variable(tf.random.normal([1]))
w2 = tf.Variable(tf.random.normal([1]))
w3 = tf.Variable(tf.random.normal([1]))
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.00001

for i in range(1000+1):
    # tf.GradientTape() to record the gradient of the cost function

    with tf.GradientTape() as tape:
        hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # calculates the gradient of the cost
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

    # update w1, w2, w3 and b
    w1.assign_sub(learning_rate * w1_grad)
    w2.assign_sub(learning_rate * w2_grad)
    w3.assign_sub(learning_rate * w3_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:12.4}".format(i, cost.numpy()))

'''
Matrix 형태로 자료 표현

multi-variable linear Regression에서 변수를 matrix 형태로 표현한 게 중요함.
'''

data = np.array([
    [ 73., 93., 89., 96., 73 ],
    [ 80., 88., 91., 98., 66 ],
    [ 75., 93., 90., 100., 70. ],
    [ 152., 185., 180., 196., 142. ]
], dtype=np.float32)

X = data[:-1, :]
Y = data[-1,:]

W =tf.Variable(tf.random.normal([1,3]))
b =tf.Variable(tf.random.normal([1]))

learning_rate = 0.00001

# hypothesis, prediction function
def predict(X):
    return tf.matmul(W,X) + b

n_epochs = 2000
for i in range(n_epochs +1):
    # record the gradient of the cost fuction
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X)-Y)))

    # calculates the gradients of the loss
    W_grad, b_grad = tape.gradient(cost, [W,b])

    # updates parameters (W and b)
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i %100 ==0:
        print("{:5} | {:10.4}".format(i,cost.numpy()))