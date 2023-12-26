import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt

stock = fdr.DataReader('KO', '2010-01-01', '2019-12-31') #KO: 코카콜라
print(stock)
prices = np.asarray(stock['Close'], dtype='float32')
y = (prices - min(prices))/(max(prices)-min(prices)) #종가 정규화

#정규화한 종가를 x_data에 window크기만큼 담고 y_data에 1개 담기
window_size = 60
data_length = len(y) - window_size
y_data = np.zeros(data_length, dtype='float32')
x_data = np.zeros((data_length, window_size, 1), dtype='float32') #3차원으로 만들
for i in range(0, data_length): #range(len(data_length)로 하지 않는 이유는 0부터 시작해야 하기 때문)
    y_data[i]=y[i+window_size]
    x_data[i]=y[i:i+window_size].reshape(window_size,1) #reshape이 필요한 이유?
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
 test_size=0.3, shuffle=False)

from tensorflow import keras
model = keras.Sequential([
    keras.layers.LSTM(30, return_sequences = True, 
                      input_shape=(window_size, 1)),
    keras.layers.LSTM(30),
    keras.layers.Dense(1, activation=None),
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=30)
pred = model.predict(x_test) 
print(pred)

pred_price = (max(prices)-min(prices))*pred + min(prices) #예측한 종가
target_price = (max(prices)-min(prices))*y_test + min(prices) #실제 종가

plt.figure()
plt.plot(target_price, label='actual')
plt.plot(pred_price, label='prediction')
plt.legend()
plt.show()

# ex13-1.py    
