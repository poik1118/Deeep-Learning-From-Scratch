# Mean Squared Error(MSE) (평균 제곱 오차)

import numpy as np

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


def Mean_squared_error(y, t):
  return 0.5 * np.sum((y - t)**2)


print(Mean_squared_error(np.array(y), np.array(t)))  # 0.09750000000000003

x = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(Mean_squared_error(np.array(x), np.array(t)))  # 0.5975
