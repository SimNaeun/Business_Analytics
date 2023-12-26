# 초기모델 설정
import pandas as pd
sd=pd.read_csv('softdrink.csv')
import tensorflow as tf
x=tf.constant(sd['temp'], dtype=tf.float32)
y=tf.constant(sd['sales'], dtype=tf.float32)
print(x)
print(y)

a=tf.Variable(20, dtype=tf.float32)
b=tf.Variable(650, dtype=tf.float32)
y_hat= a*x+b
print(y_hat)

import matplotlib.pyplot as plt
plt.scatter(x, y, color='blue')
plt.scatter(x, y_hat, color='red')
plt.show()

# 오차 측정
def y_hat(x, a, b):
    return a*x+b # 스칼라(a, b): 크기만 존재, 벡터(x, y): 크기와 방향 존재

def loss(a, b, x, y): #loss: 손실함수 즉, MSE
    return tf.reduce_mean(tf.square(y-y_hat(x, a, b)))


## 정리: a=20, b=650을 적용한 초기모델의 MSE를 계산해 출력하기
import pandas as pd
import tensorflow as tf

def y_hat(x, a, b):
    return a*x+b

def loss(a, b, x, y):
    return tf.reduce_mean(tf.square(y-y_hat(x, a, b))) #벡터의 평균 계산

sd=pd.read_csv('softdrink.csv')
x=tf.constant(sd['temp'], dtype=tf.float32)
y=tf.constant(sd['sales'], dtype=tf.float32)
a=tf.Variable(20, dtype=tf.float32)
b=tf.Variable(650, dtype=tf.float32)

print('MSE=', loss(a, b, x, y))


#최적화
with tf.GradientTape() as t: 
    current_loss=loss(a, b, x, y)
#GradientTape: 미분 계산, 변수 t에 GradientTape이 기록됨

da, db=t.gradient(current_loss[a,b])
# a와 b를 리스트로 묶음으로써 a와 b 각각의 gradient를 구함

print('da=', da)
print('db=', db)

a.assign_add(-0.001*da)
b.assign_add(-0.001*db)


