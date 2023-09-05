# MNIST Show

import numpy as np

import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져옴
# 정식 디렉터리 주소 : "/workspaces/Deeep-Learning-From-Scratch"

from mnist import load_mnist  # mnist.py파일의 load_mnist함수 호출 / Line 90
from PIL import Image  # 이미지 표시를 위한 PIL(Python Image Library) 호출


def Img_show(img):
  pil_img = Image.fromarray(np.uint8(img))
  # Image.fromarray()함수는 배열 객체를 입력으로 받아 배열 객체에서 만든 이미지 객체를 반환합니다.
  # np.uint8(img)함수는 img를 0~255까지의 정수형으로 데이터를 표현한다.
  pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                  normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784, ) / 요소 784개가 포함된 1차원 배열
img = img.reshape(28, 28)  # 원래 이미지의 모양으로 변형
print(img.shape)  # (28, 28)

Img_show(img)