from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_images.shape)
print(train_labels.shape)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
model.fit(train_images, train_labels, epochs=1)
predictions = model.predict(test_images)

import matplotlib.pyplot as plt
names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
for i in range(1):
    print('Actual : ', test_labels[i], names[test_labels[i]])
    plt.subplot(1,2,1)
    plt.imshow(test_images[i])
    plt.title('Actual : '+names[test_labels[i]])
    
    plt.subplot(1,2,2)
    plt.bar(names, predictions[i])
    plt.xticks(rotation='vertical')
    plt.show()

import numpy as np
predicted_labels=[]
for p in predictions:
    predicted_labels.append(np.argmax(p))

from sklearn.metrics import classification_report
print(classification_report(test_labels, predicted_labels))

# ex12-3.py
