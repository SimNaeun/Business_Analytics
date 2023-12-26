from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
y_true=[0,0,0,0,1,1,1,0,0,1]
y_pred=[0,0,0,1,1,1,0,0,1,1]
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

# BMI=weight/height^2
# 키를 알 수 없어 몸무게 데이터만으로 고도비만여부를 판정해야 하는 경우
weight=[58,62,69,73,77,84,88,92,96,104]
height=[160,163,151,175,178,164,165,193,172,183]
y_true=[0,0,1,0,0,1,1,0,1,1]

# threshold: 임계치
fpr_list=[]
tpr_list=[]
for threshold in range(55, 110, 5):
    y_pred=[]
    for i in range(10):
            w=weight[i]
            if w>=threshold:
                y=1
            else:
                y=0
            y_pred.append(y)
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0][0]
    tp = cm[1][1]
    fn = cm[1][0]
    fp = cm[0][1]
    tpr = tp / (tp+fn) #참 양성 비율=재현율: 실제 positive(실제값) 중 positive로 판단(=맞은 경우)한 비율
    fpr = fp / (fp+tn) #허위 양성 비율: 실제 negative(실제값) 중 negative로 판단(=틀린 경우)한 비율
    print(threshold, tpr, fpr)

    fpr_list.append(fpr)
    tpr_list.append(tpr)

import matplotlib.pyplot as plt
plt.plot(fpr_list, tpr_list, '-o')
plt.show()


weight=[58,62,69,73,77,84,88,92,96,104]
height=[160,163,151,175,178,164,165,193,172,183]
y_true=[0,0,1,0,0,1,1,0,1,1]
import numpy as np

fpr_list=[]
tpr_list=[]
for threshold in np.arange(20, 40, 0.1): # range 함수는 정수로만 증가시킬 수 있기 때문에 arrange 함수로 20부터 40까지 0.1 간격으로 촘촘히 증가
    y_pred=[]
    for i in range(10):
            bmi=weight[i]/(height[i]/100)**2 # height는 m 단위로 넣어야 하기 때문에 100으로 나눔
            if bmi>=threshold:
                y=1
            else:
                y=0
            y_pred.append(y)
    cm = confusion_matrix(y_true, y_pred)
    tn = cm[0][0]
    tp = cm[1][1]
    fn = cm[1][0]
    fp = cm[0][1]
    tpr = tp / (tp+fn) #참 양성 비율=재현율: 실제 positive(실제값) 중 positive로 판단(=맞은 경우)한 비율
    fpr = fp / (fp+tn) #허위 양성 비율: 실제 negative(실제값) 중 negative로 판단(=틀린 경우)한 비율
    print(threshold, tpr, fpr)

    fpr_list.append(fpr)
    tpr_list.append(tpr)

import matplotlib.pyplot as plt
plt.plot(fpr_list, tpr_list, '-o')
plt.show()
