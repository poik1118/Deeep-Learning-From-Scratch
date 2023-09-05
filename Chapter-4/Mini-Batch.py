# Mini-Btach (미니배치 학습)

import sys
import os
sys.path.append(os.pardir)

import numpy as np
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                  one_hot_label=True)

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

print(x_batch)
print(t_batch)

print(np.random.choice(60000, 10))


# 배치용 교차 엔트로피 오차 구현
def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  batch_size = y.shape[0]

  #return -np.sum(t * np.log(y)) / batch_size
  return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size