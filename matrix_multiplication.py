import tensorflow as tf
a = tf.constant(1)
print(a)

a = tf.constant(1, dtype=tf.float32)
print(a)

a=tf.constant([[3.14, 1.7], [5.6, 2.2]], dtype=tf.float32)
print(a)

a=tf.constant([[1,2], [3,4]], dtype=tf.float32)
b=tf.constant([[2,2], [3,5]], dtype=tf.float32)
print(a+b) #행렬의 합
print(tf.matmul(a,b)) #matrix multiplication: 행렬의 곱

a=tf.constant([[1,2], [3,4]], dtype=tf.float32)
i=tf.constant(tf.eye(2), dtype=tf.float32)
print(tf.matmul(a, i))
print(tf.matmul(i, a))

aInv=tf.linalg.inv(a)
print(aInv)
print(tf.matmul(a, aInv))


x=[30, 20, 50, 40, 10]
y=[120, 60, 140, 100, 70]
import statsmodels.api as sm
results= sm.OLS(y, sm.add_constant(x)).fit()
print(results.summary())

