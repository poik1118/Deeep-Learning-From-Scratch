# 3rd Nueral Network (3층 신경망)

import numpy as np

def Sigmoid(x):
  return 1 / (1 + np.exp(-x))


## 0층(입력층) -> 1층(은닉층) 과정 구현
print("### 0층 -> 1층 ###")
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
Z1 = Sigmoid(A1)

print(A1)
print(Z1)

## 1층(은닉층) -> 2층(은닉층) 과정 구현
print("\n### 1층 -> 2층 ###")
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = Sigmoid(A2)

print(A2)
print(Z2)

## 2층(은닉층) -> 3층(출력층) 과정 구현
print("\n### 2층 -> 3층###")


def Identity_funtion(x):  # 항등함수(입력을 그대로 출력) 정의
  return x


W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

print(W3)
print(B3)

A3 = np.dot(Z2, W3) + 3
Y = Identity_funtion(A3)  # or Y = A3

print(A3)
print(Y)