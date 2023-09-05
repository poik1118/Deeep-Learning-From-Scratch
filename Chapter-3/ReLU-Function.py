# ReLU Function
# 입력이 0을 넘으면 그 입력을 그대로 출력하며, 입력이 0이하면 0을 출력하는 함수

import numpy as np
import matplotlib.pylab as plt

def Relu(x):
  return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = Relu(x)

plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()
