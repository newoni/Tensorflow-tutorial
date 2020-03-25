import numpy as np
import tensorflow as tf

'''
logistic regression 이후, 
softmax , one-hot encoding.

softmax를 사용했기 때문에 모든 결과값들이 0~1의 값 -> cross entropy의 -log 안에 들어갈 수 있음.

softmax, one-hot encoding, cross entropy 의 관계에 대해서 잘 생각해보기 
'''

x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]

y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

nb_classes = 3  #num classes

W = tf.Variable(tf.random.normal([4, nb_classes]),name='weight')
b = tf.Variable(tf.random.normal([nb_classes]),name='bias')
variables = [W,b]
def hypothesis(X):
    return tf.nn.softmax(tf.matmul(X,W) + b)


# Softmax onehot test
sample_db = [[8,2,1,4]]
sample_db = np.asarray(sample_db, dtype=np.float32)

print(hypothesis(sample_db))

# Cost function
def cost_fn(X,Y):
    logits = hypothesis(X)
    cost = -tf.reduce_sum(Y*tf.math.log(logits), axis =1)
    cost_mean = tf.reduce_mean(cost)
    return cost_mean

def grad_fn(X, Y):
    with tf.GradientTape() as tape:
        cost = cost_fn(X,Y)
        grads = tape.gradient(cost, variables)
        return grads

def fit(X, Y, epochs=2000, verbose=100):
    optimizer = tf.optimizers.Adam(learning_rate = 0.01)
    for i in range(epochs):
        grads = grad_fn(X,Y)
        optimizer.apply_gradients(zip(grads, variables))
        if (i==0) | ((i+1)%verbose==0):
            print('Loss at epoch %d: %f' %(i+1, cost_fn(X,Y).numpy()))

# a = hypothesis(x_data)

print(hypothesis)
grad_fn(x_data,y_data)
fit(x_data,y_data)

#Prediction Check
sample_data = [[2,1,3,2]]   # answer_label [[0,0,1]]
sample_data = np.asarray(sample_data, dtype=np.float32)

a= hypothesis(sample_data)

print(a)
print(tf.argmax(a,1))