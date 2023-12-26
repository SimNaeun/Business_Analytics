from tensorflow import keras
from tensorflow.keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data()
print(type(train_data))

print(train_data.shape, train_labels.shape)
print(test_data.shape, test_labels.shape)
print('Train data 0 = ', train_data[0])
print('Train label 0 = ', train_labels[0])

word_index = imdb.get_word_index()

print(word_index['the'], word_index['and'], word_index['a'])
numToWord = {}
numToWord[0]=''
numToWord[1]='<시작>'
numToWord[2]='<미등록>'
numToWord[3]=''
for (word, num) in word_index.items():
    numToWord[num+3] = word
print(numToWord[4], numToWord[5], numToWord[6])

for i in train_data[0]: print(numToWord[i], end=' ')    

#빈도 상위 1000개 데이터만 가져오기
#빈도 수가 1000 초과이면 세지 않음
def vectorize(review_data):
    vec=np.zeros(1000, dtype='float32')
    for x in review_data:
        if x>=1000: continue
        if vec[x]<1: vec[x]+=0.25 #빈도가 4번 이상이 나오면 다 1로 만듦
    return vec

print(vectorize(train_data[0]))

train_input = np.zeros((len(train_data), 1000))
for i, review in enumerate(train_data):
    train_input[i] = vectorize(review)

model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(1000,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(train_input, train_labels, epochs=3)


test_input = np.zeros((len(test_data), 1000))
for i, review in enumerate(test_data):
    test_input[i]=vectorize(review)

pred = model.predict(test_input)
print(pred)

pred_labels=[]
for v in pred:
    if v>=0.5: pred_labels.append(1)
    else: pred_labels.append(0)

from sklearn.metrics import classification_report
print(classification_report(test_labels, pred_labels))

# ex14-5.py    
