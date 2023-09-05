# Stair Function (계단 함수) / 임계값을 경계로 출력이 바뀌는 함수
# 입력이 0을 넘으면 1을 출력하고, 그 외에는 0을 출력하는 코드

import numpy as np
import matplotlib.pylab as plt

def StepFunction(x):
  return np.array(x > 0, dtype=np.int)


X = np.arange(-5.0, 5.0, 0.1)
Y = StepFunction(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()