import numpy as np
from tensorflow import keras
from sklearn import datasets

dataset = datasets.load_digits()
x_data = (dataset.data/15.0).astype(np.float32)
y_data = (dataset.target %2).astype(np.float32)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

model = keras.Sequential([
    keras.layers.Dense(2, activation='sigmoid', input_shape=(64,)),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='SGD', loss='binary_crossentropy')
model.fit(x_train, y_train, epochs=200)

print(loss)

predictions = model.predict(x_test)
predLabels=[]
for y in predictions:
    if y<0.5: predLabels.append(0)
    else: predLabels.append(1)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(y_test, predLabels))
print(classification_report(y_test, predLabels))


#출력이 0 또는 1이 아니라 0~9가 나오도록
import numpy as np
from tensorflow import keras
from sklearn import datasets

dataset = datasets.load_digits()
x_data = (dataset.data/16.0).astype(np.float32)
y_data = (dataset.target).astype(np.float32)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

model = keras.Sequential([
    keras.layers.Dense(2, activation='relu', input_shape=(64,)),
    keras.layers.Dense(1, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=1000)



