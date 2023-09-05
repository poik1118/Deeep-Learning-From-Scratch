# MNIST 신경망의 추론 처리, 배치처리 적용

import numpy as np

import pickle
import mnist


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def SoftMax(ary):
  aryMax = np.max(ary)
  exp_ary = np.exp(ary - aryMax)
  sum_exp_ary = np.sum(exp_ary)
  result = exp_ary / sum_exp_ary

  return result

def Get_data():
  (x_train, t_train), (x_test, t_test) = \
    mnist.load_mnist(normalize=True, flatten=True, one_hot_label=False)

  return x_test, t_test

def Init_network():
  with open("GithubSources/sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

  return network

x, t = Get_data()
network = Init_network()

batch_size = 100  # 배치 크기 지정
accuracy_cnt = 0

def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = Sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = Sigmoid(a2)
  a3 = np.dot(z2, W3) + b3
  y = SoftMax(a3)

  return y


for i in range(0, len(x), batch_size):  #
  x_batch = x[i:i + batch_size]
  y_batch = predict(network, x_batch)
  p = np.argmax(y_batch, axis=1)
  accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy : " + str(float(accuracy_cnt) / len(x)))  # 0.9352

# 배치 이론
x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3],
              [0.8, 0.1, 0.1]])
y = np.argmax(x, axis=1)
print(y)  # [1 2 1 0]

y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y == t)  # [ True  True False  True]
print(np.sum(y == t))  # 3