from sklearn import datasets
dataset = datasets.load_digits()
x_data = dataset.data
y_data = dataset.target

print('X shape = ', x_data.shape)
print('Y shape = ', y_data.shape)

print('First x data = ', x_data[0])
print('First y data = ', y_data[0])


import matplotlib.pyplot as plt
plt.imshow(x_data[0].reshape(8,8))
plt.colorbar()
plt.show()

# ex 11-1.py
