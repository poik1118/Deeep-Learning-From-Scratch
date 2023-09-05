# Multi Demotional Arrays (다차원 배열)

import numpy as np

A = np.array([1, 2, 3, 4])
print(A)
print(str(np.ndim(A)) + "차원 배열\n")  # 현재 N차원 배열임을 나타냄

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(str(np.ndim(B)) + "차원 배열\n")

C = np.array([[[1, 2], [3, 4], [5, 6]], [[10, 20], [30, 40], [50, 60]]])
print(C)
print(str(np.ndim(C)) + "차원 배열\n")
