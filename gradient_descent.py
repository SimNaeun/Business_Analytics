import pandas as pd
import tensorflow as tf

def y_hat(x, a, b):
    return a*x + b

def loss(a, b, x, y):
    return tf.reduce_mean(tf.square(y - y_hat(x, a, b)))

sd = pd.read_csv('softdrink.csv')
x = tf.constant(sd['temp'], dtype=tf.float32)
y = tf.constant(sd['sales'], dtype=tf.float32)
a = tf.Variable(20, dtype=tf.float32)
b = tf.Variable(650, dtype=tf.float32)

while True:
    with tf.GradientTape() as t:
        current_loss=loss(a, b, x, y)
    da, db = t.gradient(current_loss, [a,b])
    print('MSE, a, b, da, db = ', 
        loss(a,b,x,y).numpy(), a.numpy(), b.numpy(), da.numpy(), db.numpy())

    a.assign_add(-0.001*da)
    b.assign_add(-0.001*db)

    if abs(da)<0.05 and abs(db)<0.05: break

import matplotlib.pyplot as plt
plt.scatter(x, y, color='blue')
plt.plot(x, y_hat(x,a,b), color='red')
plt.show()

# ex9-4.py

