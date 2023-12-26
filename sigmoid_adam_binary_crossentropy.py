import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split

def output(w,x,b):
    z = w*x + b
    a = 1/(1+tf.math.exp(-z))
    return a    

def loss(w,b,x,y):
    a = output(x,w,b)
    l = y*tf.math.log(a) + (1-y)*(tf.math.log(1-a))
    return -tf.reduce_mean(l)

dataset = datasets.load_digits()
print(type(dataset.target[0]))
x_data = (dataset.data/15.0).astype(np.float32)
y_data = (dataset.target %2).astype(np.float32)

print(type(y_data[0]))

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)


# import matplotlib.pyplot as plt
# for i in range(6):
#     plt.imshow(x_train[i].reshape(8,8))
#     plt.colorbar()
#     plt.title("Target=%s"%y_train[i])
#     plt.show()


model = keras.Sequential([
    keras.layers.Dense(2, activation='sigmoid', input_shape=(64,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

predictions = model.predict(x_test)
predLabels=[]
for y in predictions:
    if y<0.5: predLabels.append(0)
    else: predLabels.append(1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(y_test, predLabels))
print(classification_report(y_test, predLabels))



exit(0)
import matplotlib.pyplot as plt
plt.imshow(dataset.data[0].reshape(8,8))
plt.colorbar()
plt.show()
