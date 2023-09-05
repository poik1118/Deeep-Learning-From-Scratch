# Matrix Multiplication (행렬 곱)
## Inner Neural Network (신경망의 내적)

import numpy as np

X = np.array([1, 2])
print(X.shape)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)

Y = np.dot(X, W)
print(Y)
# 편향과 활성화 함수를 생략하고 가중치만 갖는다.
# X의 1번째 차원 = W의 0번째 차원, W의 1번째 차원 = Y의 0번째 차원.