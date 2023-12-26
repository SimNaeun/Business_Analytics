import pandas as pd
df = pd.read_csv('bank_customer.csv')
feature = df[['Customer_Age', 'Gender', 'Dependent_count', 'Income_Category',
  'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 
  'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 
  'Avg_Open_To_Buy', 'Total_Trans_Amt', 'Total_Amt_Chng_Q4_Q1']]
target = df['Attrition_Flag']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=2222)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=1111) # 아직 데이터를 넣지 않아 학습되지 않은 깡통 모델
dtc = dtc.fit(x_train,y_train) #학습된 모델
y_pred = dtc.predict(x_test)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#관심사는 0이 아니라 1, 1을 보면 precision:0.81 , recall:0.77 , f1-score:0.79
#이탈 고객을 놓침( cm[1][0] ): 79
#이탈 고객이 아닌데 이탈고객이라고 판단( cm[0][1] ): 62
#이탈고객(1): 343 = 264+79

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(24,10))  # figsize는 생략 가능
plot_tree(dtc, feature_names=['Customer_Age', 'Gender', 
  'Dependent_count', 'Income_Category',
  'Card_Category', 'Months_on_book', 'Total_Relationship_Count', 
  'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 
  'Avg_Open_To_Buy', 'Total_Trans_Amt', 'Total_Amt_Chng_Q4_Q1'], 
  class_names=['Existing','Attrited'], max_depth=4, rounded=True, fontsize=6)
plt.show()
