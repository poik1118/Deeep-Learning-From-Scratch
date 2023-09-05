# Cross Entropy Error(CEE) (교차 엔트로피 오차)

import numpy as np

def Cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(t * np.log(y + delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(Cross_entropy_error(np.array(y), np.array(t)))  # 0.510825457099338

x = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(Cross_entropy_error(np.array(x), np.array(t)))  # 2.302584092994546
