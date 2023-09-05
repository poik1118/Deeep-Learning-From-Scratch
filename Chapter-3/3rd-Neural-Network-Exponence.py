# # 3rd Nueral Network Exponence (3층 신경망 구현정리)

import numpy as np

def Sigmoid(x):
  return 1 / (1 + np.exp(-x))

def Identity_funtion(x):
  return x

def Init_network():
  network = {}
  network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])

  return network


def Forward(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = Sigmoid(a1)

  a2 = np.dot(z1, W2) + b2
  z2 = Sigmoid(a2)

  a3 = np.dot(z2, W3) + b3
  y = Identity_funtion(a3)

  return y


network = Init_network()

x = np.array([1.0, 0.5])
y = Forward(network, x)

print(y)