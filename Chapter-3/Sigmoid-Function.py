# Sigmoid Function (시그모이드 함수)
# 함수의 반환값은 단조증/감하고, 시그모이드 함수의 반환값(y축)은 보통 0에서 1까지의 범위를 가진다.

import numpy as np
import matplotlib.pylab as plt

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


X = np.arange(-5.0, 5.0, 0.1)
Y = Sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()