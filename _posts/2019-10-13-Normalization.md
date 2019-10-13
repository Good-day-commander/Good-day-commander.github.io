---
layout: post
title: "Normalization"
description: An example post which explain why we use nomalization to data.
date: 2019-10-13 02:47
categories: Tensorflow JupytorNotebook
---
### 언제 *Normalization*이 필요한가?


보통 데이터를 통해 학습할 때, 항목에 따라 데이터 값의 크기가 차이 나는 경우가 대부분이다.

예를 들어 보자. x1항목은 한자리 수 값들인 반면 x2는 네자리 수 값들이다. 그렇다면 y는 당연히 x1항목보다 x2항목의 값에 민감하게 변화할 수 밖에 없다.

이러한 편향을 해결하기 위해, '*Nomalization*'을 활용한다.

Normalization을 써보기 이전에, 어떤 상황이 Normalization이 필요한 상황인지 알아보자.

- - -

~~~python
xy = np.array([[831.05499, 908100, 828.349976, 831.659973],
                [825.54504, 1828100, 821.655029, 828.070007],
                [822.16504, 1438100, 818.97998, 824.159973],
                [818.47949, 1008100, 815.48999, 819.23999],
                [821.17999, 1188100, 818.469971, 818.97998],
                [821.0000, 1198100, 816, 820.450012],
                [813.47498, 1098100, 809.780029, 813.669983],
                [813.08496, 1398100, 804.539978, 809.559998]], dtype = np.float32)
~~~
x1, x3의 스케일에 비해 x2가 유독 큰 것을 확인할 수 있다.
x2는 n X 10^6 
x1과 x3는 n X 10^2

자료들을 플로팅하면 좀 더 극명하게 알 수 있다.
~~~python
plt.figure(1)
plt.scatter(x_train[:,0], x_train[:,1])
plt.axis('equal')
plt.xlabel('x1')
plt.ylabel('x2')

plt.figure(2)
plt.scatter(x_train[:,1], x_train[:,2])
plt.axis('equal')
plt.xlabel('x2')
plt.ylabel('x3')

plt.figure(3)
plt.scatter(x_train[:,0], x_train[:,2])
plt.axis('equal')
plt.xlabel('x1')
plt.ylabel('x3')

plt.show()
~~~

![no_normalization_flotting](/images/flotting1.PNG)

x2와 비교하게 되면 x1과 x3는 거의 차이가 드러나지 않게 되는 상황.
(x1과 x3끼리 비교시에는 차이가 잘 드러나는 편)


이 상황에서 Normalization하지 않고 그대로 학습을 진행하게 된다면
Loss가 NAN으로 나오게 되는 참사 발생.

~~~python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
tf.enable_eager_execution()
tf.set_random_seed(777)

xy = np.array([[831.05499, 908100, 828.349976, 831.659973],
[825.54504, 1828100, 821.655029, 828.070007],
[822.16504, 1438100, 818.97998, 824.159973],
[818.47949, 1008100, 815.48999, 819.23999],
[821.17999, 1188100, 818.469971, 818.97998],
[821.0000, 1198100, 816, 820.450012],
[813.47498, 1098100, 809.780029, 813.669983],
[813.08496, 1398100, 804.539978, 809.559998]], dtype = np.float32)

x_train = xy[:, 0:-1]
y_train = xy[:, [-1]]

W = tf.Variable(tf.random_normal([3,1]))
b = tf.Variable(tf.random_normal([1]))

def lin_reg(X):
    hypothesis = tf.matmul(X,W) + b
    return hypothesis

def cost_fn(hypothesis, Y):
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    return cost

def grad(X, Y):
    with tf.GradientTape() as tape:
        loss_value = cost_fn(lin_reg(X), Y)
    return tape.gradient(loss_value, [W, b])

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

EPOCHS = 1001

for step in range(EPOCHS):
    X = tf.cast(x_train, tf.float32)
    Y = tf.cast(y_train, tf.float32)
    
    grads = grad(X,Y)
    optimizer.apply_gradients(grads_and_vars=zip(grads, [W,b]))
    
    if step % 100 == 0:
        print("lter: {:5}, Loss: {:.4f}".format(step, cost_fn(lin_reg(X), Y)))
~~~
![no_normalization](/images/no_normalization.PNG)


- - -
### 어떻게 *Normalization*할까?

![normalization](/images/how_to_normalization.PNG)

사진 설명이 워낙 명료해서 굳이 코멘트를 붙일 필요 없을 듯

코드는 이렇게
~~~python
def normalization(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / denominator

xy = normalization(xy)
~~~
![normalization](/images/result_normalization.png)
깔끔하게 [0, 1] 범위 값들로 정리된 것을 확인할 수 있다.

이 코드를 X_train, y_train 정의 전에 삽입 후 결과를 살펴보자.


![normalization](/images/result2_normalization.PNG)
![normalization](/images/result3_normalization.PNG)


플로팅 결과는 이전에 비해 훨씬 값들의 관계를 잘 나타나게 되었으며,
결과 또한 Loss가 NAN으로 발산하는 것이 아니라 수렴하는 것을 확인 할 수 있음.


- - -

정리
1. 데이터 값들간의 크기차이가 나게 되면 문제 발생
2. *Normalization* 하여 자료를 Data를 scailing 해준다면 해결!
