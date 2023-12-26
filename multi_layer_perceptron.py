import numpy as np
from sklearn import datasets


def output(x, w, b):
    z = b
    for j in range(len(x)):
        z += w[j]*x[j]
    a = 1/(1+np.exp(-z))
    return a    

def loss(a, y):
    l = y*np.log(a) + (1-y)*(np.log(1-a))
    return -np.mean(l)

def predict(x, w1,b1,w2,b2,w31,w32,b3):
    n = len(x)
    a = np.zeros(n)
    for i in range(n):
        a1 = output(x[i], w1, b1)
        a2 = output(x[i], w2, b2)
        a3 = output([a1, a2], [w31, w32], b3)
        a[i] = a3
    return a


dataset = datasets.load_digits()
x_data = dataset.data
y_data = dataset.target
x_data = (x_data / 15.0).astype(np.float32)
y_data = (y_data % 2).astype(np.float32)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
n = len(y_train)

w1 = np.zeros(64)
b1 = 0
w2 = np.zeros(64)
b2 = 0
w31 = 0
w32 = 0
b3 = 0

eta = 0.1

for epoch in range(200):
    dw1 = np.zeros(64)
    db1 = 0
    dw2 = np.zeros(64)
    db2 = 0
    dw31 = 0
    dw32 = 0
    db3 = 0

    a = predict(x_train, w1,b1,w2,b2,w31,w32,b3)
    print('loss = ', loss(a, y_train))
    for i in range(n):
        x = x_train[i]
        y = y_train[i]
        a1 = output(x, w1, b1)
        a2 = output(x, w2, b2)
        a3 = output([a1, a2], [w31, w32], b3)

        dw31 += (a3-y)*a1
        dw32 += (a3-y)*a2
        db3  += (a3-y)

        for j in range(64):
            dw1[j] += (a3-y)*w31*a1*(1-a1)*x[j]
        db1 += (a3-y)*w31*a1*(1-a1)

        for j in range(64):
            dw2[j] += (a3-y)*w32*a2*(1-a2)*x[j]
        db2 += (a3-y)*w32*a2*(1-a2)

    dw31/=n
    dw32/=n
    db3/=n
    dw1/=n
    db1/=n
    dw2/=n
    db2/=n

    w1 -= eta*dw1
    b1 -= eta*db1
    w2 -= eta*dw2
    b2 -= eta*db2
    w31 -= eta*dw31
    w32 -= eta*dw32
    b3 -= eta*db3


predictions = predict(x_test, w1,b1,w2,b2,w31,w32,b3)
predLabels=[]
for y in predictions:
    if y<0.5: predLabels.append(0)
    else: predLabels.append(1)

print(w1)
print('b1=',b1)
print(w2)
print('b2=',b2)
print(w31,w32,b3)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(y_test, predLabels))
print(classification_report(y_test, predLabels))



# exit(0)
# import matplotlib.pyplot as plt
# plt.imshow(dataset.data[0].reshape(8,8))
# plt.colorbar()
# plt.show()
