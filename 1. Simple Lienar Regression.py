'''
Hypothesis (가설함수)
H(x) = Wx + b

Cost (손실함수)
cost(W,b) = 1/m * sigma((H(x)-y)**2)

Cost값이 최소가 되는 W,b 값을 찾는 것 -> learning 이라고 함.
'''

import tensorflow as tf

x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]

W = tf.Variable(2.9)
b = tf.Variable(0.5)

# hypothesis = W * x + b
hypothesis = W * x_data + b

cost = tf.reduce_mean(tf.square(hypothesis-y_data))

'''
cf
reduce_mean, square 연습
'''
v = [1.,2.,3.,4.]
print(tf.reduce_mean(v))

print(tf.square(3))
#####################

'''
Gradient descent
'''

# learning_rate initialize
learning_rate = 0.01

for i in range(100):
    # Gradient decent
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    W_grad, b_grad = tape.gradient(cost, [W,b])

    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)

    if i% 10 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i,W.numpy(), b.numpy(), cost))

#Predict
print(W*5 +b)
print(W*2.5 +b)

'''
sklearn 으로 linear regression 표현
'''

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


print("\n**********************************************\n"
      "sklearn 으로 linear regression 표현\n"
      "**********************************************\n")

regr = LinearRegression()

x_data = [1,2,3,4,5]
y_data = [2,4,6,8,10]
x_data_arr, y_data_arr = np.array(x_data), np.array(y_data)
x_data_arr= x_data_arr.reshape(-1,1)
y_data_arr= y_data_arr.reshape(-1,1)

regr.fit(x_data_arr,y_data_arr)

predictions = regr.predict(np.array([[5],[2.5]]))

print(predictions)

print("MSE: {:5.2f}".format(mean_squared_error(np.array([[10],[5]]),predictions)))
