# Matrix Multiplication (행렬 곱)
## Two-dimensional Array (2차원 배열)

import numpy as np

A = np.array([[1, 2], [3, 4]])
print(A.shape)

B = np.array([[5, 6], [7, 8]])
print(B.shape)

Dot = np.dot(A, B)
print(Dot)



A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)

print(np.dot(A, B))
# 2X3 & 3X2 ==> 2X2 형태의 배열이 됨

C = np.array([[1, 2], [3, 4]])
print(C.shape)
print(np.dot(A, C))
# shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)
# A의 1번째 차원과 C의 0번쨰 차원의 원소 수가 다르다. / A = 2X3  !=  C = 2X2
# 다차원 배열을 곱하려면 두 행렬의 대응하는 차원의 원소 수를 일치시켜야함.



A = np.array([[1, 2], [3, 4], [5, 6]])
print(A.shape)

B = np.array([7, 8])
print(B.shape)

print(np.dot(A, B))
# A가 3X2로 2차원 행렬이고 B가 2개의 원소인 1차원 배열임.
# A의 1번째 차원과 B의 0번째 차원의 원소 수가 같으므로,
# 대응하는 차원의 원소 수가 일치한다는 원칙이 성립함.