f = open('정보화기본계획.txt', 'r', encoding='utf8')
texts=[]
while True:
    line = f.readline()
    if not line: break
    texts.append(line)
text = ' '.join(texts)
special = ['\n', '.', ',', '-', ':', '~', '/', '·', '●', '▲', '※', 
           '(', ')', '[', ']', '‘', '’', '→']
for c in special:
    text = text.replace(c, ' ')
wordList = text.split(' ')

stopwords=['등', '및', '위한', '통해', '등을', '통한', '활용', '위해', 
    '수', '있는', '필요', '대한', '관련', '있도록', '다양한', 
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'] 
words = {}
for w in wordList:
    if len(w)==0: continue
    if w in stopwords: continue
    if w not in words: words[w] = 1
    else: words[w] += 1

sortedWords = sorted(words.items(), key=(lambda x: x[1]), reverse=True)
print(sortedWords)

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(font_path='./KoPubWorld Dotum Medium.ttf',
    background_color='white', width=1024, height=1024,
    max_font_size=100, min_font_size=10, max_words=300, 
    colormap='Dark2').generate_from_frequencies(words)

#mask를 바꾸고 싶을 때 사용하는 코드
#import numpy as np
#from PIL import Image # pip install image
#maskPic = np.array(Image.open("mask.png"))
#wordcloud = WordCloud(font_path='./KoPubWorld Dotum Medium.ttf',
#    background_color='white', width=1024, height=1024,
#    max_font_size=100, min_font_size=10, max_words=300, 
#    colormap='Dark2', mask=maskPic).generate_from_frequencies(words)
plt.figure()
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# ex14-2.py
