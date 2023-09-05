# MNIST 배치처리과정

import pickle
from mnist import load_mnist

def Get_data():
  (x_train, t_train), (x_test, t_test) = \
    mnist.load_mnist(normalize=True, flatten=True, one_hot_label=False)

  return x_test, t_test


def Init_network():
  with open("GithubSources/sample_weight.pkl", 'rb') as f:
    network = pickle.load(f)

  return network

x, _ = Get_data()
network = Init_network()

W1, W2, W3 = network['W1'], network['W2'], network['W3']

print(x.shape)  # (10000, 784)
print(W1.shape)  # (784, 50)
print(W2.shape)  # (50, 100)
print(W3.shape)  # (100, 10)