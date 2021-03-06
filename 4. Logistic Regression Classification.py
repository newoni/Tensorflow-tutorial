import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
logistic_regression 함수를 보면

W*x + b 의 형태에서 sigmoid 함수를 넣어준 형태.
hypothesis를 한번에 선언함.

loss_fn의 경우, cross entropy 사용.

grad 함수에서 logistic_regression과 loss_fn을 함께 엮음.

이후 마지막 for문이 main 문
'''

x_train =[[1.,2.],
          [2.,3.],
          [3.,1.,],
          [4.,3.,],
          [5.,3.],
          [6.,2.,]]

y_train =[[0.],
          [0],
          [1.],
          [1.],
          [1.],
          [1.]]

x_test = [[5.,2.,]]
y_test = [[1.]]

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))

W = tf.Variable(tf.zeros([2,1]), name ='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

def logistic_regression(features):
    hypothesis = tf.divide(1.,1.+ tf.exp(tf.matmul(features,W)+b))
    return hypothesis

def loss_fn(hypothesis, features, labels):
    cost = -tf.reduce_mean(labels*tf.math.log(logistic_regression(features)) + (1-labels)*tf.math.log(1-hypothesis))
    return cost

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy

def grad(features, labels):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(logistic_regression(features),features,labels)
    return tape.gradient(loss_value, [W,b])

EPOCHS = 1001
for step in range(EPOCHS):
    for features, labels in iter(dataset):
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars = zip(grads,[W,b]))

        if step%100 == 0:
            print("Iter: {} | Loss: {:.4f}".format(step, loss_fn(logistic_regression(features),features,labels)))

    test_acc = accuracy_fn(logistic_regression(x_test),y_test)