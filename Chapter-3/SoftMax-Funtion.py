# SoftMax Funtion (소프트맥스 함수)
## numpy.exp(a)의 exp는 자연상수 e(약 2.718) 와 a승으로 계산한다. (e ^ a)

import numpy as np
import matplotlib.pylab as plt


a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)  # exp_a = [e^0.3, e^2.9, e^4.0]
print(exp_a)

sum_exp_a = np.sum(exp_a)  # 지수 함수의 합
print(sum_exp_a)

result = exp_a / sum_exp_a  # [1.3498/74.122, 18.174/74.122, 54.598/74.122]
print(result)


# SoftMax함수 구현의 주의점
ary = np.array([1010, 1000, 990])
div = np.exp(ary) / np.sum(np.exp(ary))  # 기존 SoftMax함수 계산
print(div)  # 계산오류

max_ary = np.max(ary)  # ary의 최댓값 = 1010
minus_ary = ary - max_ary  # 배열과 최댓값이 차
print(minus_ary)

result = np.exp(ary - max_ary) / np.sum(np.exp(ary - max_ary))  # 변경된 SoftMax함수
print(result)


# SoftMax함수 정의
def softMax(ary):
  aryMax = np.max(ary)
  exp_ary = np.exp(ary - aryMax)  # Overflow 예방
  sum_exp_ary = np.sum(exp_ary)
  result = exp_ary / sum_exp_ary

  return result


# 소프트맥스 함수의 출력은 0 ~ 1.0 사이의 실수이다. 그러면 출력의 총합은 1이 된다.
# 이 뜻은 소프트맥스 함수의 값을 확률로 사용 가능하다는 것이다.
x = np.array([0.3, 2.9, 4.0])
y = softMax(x)

print(y)
print(np.sum(y))


# SoftMax function Graph
x = np.arange(-5.0, 5.0, 0.1)
y = softMax(x)

fig = plt.figure(figsize=(10, 7))
fig.set_facecolor('white')

plt.plot(x, y)
plt.ylim(0, 0.1)
plt.title("Softmax Function Graph", fontsize=20)
plt.show()
