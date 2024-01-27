import numpy as np

def tf(w, d):
    cnt = 0
    for t in d:
        if t==w: cnt+=1
    return cnt

def df(w, D):
    cnt = 0
    for sent in D:
        if w in sent: cnt+=1
    if cnt==0: cnt = 1
    return cnt

def tfidf(w,d,D):
    N=len(D)
    return tf(w,d)*np.log(N/df(w,D))

def magnitude(v):
    m=0
    for x in v: m += x**2
    return (np.sqrt(m))

def similarity(a,b):
    dot = np.dot(a,b)
    ma = magnitude(a)
    mb = magnitude(b)
    return(dot/(ma*mb))

sent_A = '이 영화 완전 강추 진짜 다른 영화'.split(' ')
sent_B = '진짜 강추 완전 좋은 영화'.split(' ')
sent_C = '이 영화 영화 자막 완전 진짜 완전 엉망'.split(' ')
corpus=[sent_A, sent_B, sent_C]
print(corpus)

for w in sent_A:
    print(w, tfidf(w, sent_A, corpus))

for w in sent_B:
    print(w, tfidf(w, sent_B, corpus))

for w in sent_C:
    print(w, tfidf(w, sent_C, corpus))

wordSet = ['이','영화','완전','강추','진짜','다른','좋은','자막','엉망']
tfidf_a=[]
tfidf_b=[]
tfidf_c=[]
for w in wordSet: tfidf_a.append(tfidf(w,sent_A,corpus))
for w in wordSet: tfidf_b.append(tfidf(w,sent_B,corpus))
for w in wordSet: tfidf_c.append(tfidf(w,sent_C,corpus))

print('tfidf_a = ', tfidf_a)
print()
print('tfidf_b = ', tfidf_b)
print()
print('tfidf_c = ', tfidf_c)

print('Magnitude of tfidf_a = ', magnitude(tfidf_a))
print('Magnitude of tfidf_b = ', magnitude(tfidf_b))
print('Magnitude of tfidf_c = ', magnitude(tfidf_c))
print('similarity(tfidf_a, tfidf_b) = ', similarity(tfidf_a, tfidf_b))
print('similarity(tfidf_a, tfidf_c) = ', similarity(tfidf_a, tfidf_c))
print('similarity(tfidf_b, tfidf_c) = ', similarity(tfidf_b, tfidf_c))

# ex14-4.py
