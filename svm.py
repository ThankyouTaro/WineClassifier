import pandas as pd
import numpy as np
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

df = pd.read_csv('WineData_Small.csv')
Train_X, Test_X, Train_Y, Test_Y = [], [], [], []
count = 0

#df = pd.read_csv('WineData_1.csv')
Train_X, Test_X, Train_Y, Test_Y = train_test_split(df[['type', 'ela', 'grapes', 'harmon', 'body', 'acid', 'code', 'region', 'winery', 'abv']], df['rating_1'], test_size=0.05)
SVM1 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
#SVM1 = svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto') #switch the kernel to other type
SVM1.fit(Train_X, Train_Y)
predictions_SVM = SVM1.predict(Test_X)
score_1 = accuracy_score(predictions_SVM, Test_Y)*100
print("SVM Accuracy Score -> ",score_1)
#Get the importance for features
feature_importance = np.abs(SVM1.coef_)
sorted_indices = np.argsort(feature_importance)[0]
for i in reversed(sorted_indices):
    print(f"Feature {i}: Importance = {feature_importance[0][i]}")

#df = pd.read_csv('WineData_2.csv')
Train_X, Test_X, Train_Y, Test_Y = train_test_split(df[['type', 'ela', 'grapes', 'harmon', 'body', 'acid', 'code', 'region', 'winery', 'abv']], df['rating_2'], test_size=0.05)
SVM2 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM2.fit(Train_X, Train_Y)
predictions_SVM = SVM2.predict(Test_X)
score_2 = accuracy_score(predictions_SVM, Test_Y)*100
print("SVM Accuracy Score -> ",score_2)
#Get the importance for features
feature_importance = np.abs(SVM2.coef_)
sorted_indices = np.argsort(feature_importance)[0]
for i in reversed(sorted_indices):
    print(f"Feature {i}: Importance = {feature_importance[0][i]}")

#df = pd.read_csv('WineData_3.csv')
Train_X, Test_X, Train_Y, Test_Y = train_test_split(df[['type', 'ela', 'grapes', 'harmon', 'body', 'acid', 'code', 'region', 'winery', 'abv']], df['rating_3'], test_size=0.05)
SVM3 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM3.fit(Train_X, Train_Y)
predictions_SVM = SVM3.predict(Test_X)
score_3 = accuracy_score(predictions_SVM, Test_Y)*100
print("SVM Accuracy Score -> ",score_3)
#Get the importance for features
feature_importance = np.abs(SVM3.coef_)
sorted_indices = np.argsort(feature_importance)[0]
for i in reversed(sorted_indices):
    print(f"Feature {i}: Importance = {feature_importance[0][i]}")

#df = pd.read_csv('WineData_4.csv')
Train_X, Test_X, Train_Y, Test_Y = train_test_split(df[['type', 'ela', 'grapes', 'harmon', 'body', 'acid', 'code', 'region', 'winery', 'abv']], df['rating_4'], test_size=0.05)
SVM4 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM4.fit(Train_X, Train_Y)
predictions_SVM = SVM4.predict(Test_X)
score_4 = accuracy_score(predictions_SVM, Test_Y)*100
print("SVM Accuracy Score -> ",score_4)
#Get the importance for features
feature_importance = np.abs(SVM4.coef_)
sorted_indices = np.argsort(feature_importance)[0]
for i in reversed(sorted_indices):
    print(f"Feature {i}: Importance = {feature_importance[0][i]}")


#df = pd.read_csv('WineData_5.csv')
Train_X, Test_X, Train_Y, Test_Y = train_test_split(df[['type', 'ela', 'grapes', 'harmon', 'body', 'acid', 'code', 'region', 'winery', 'abv']], df['rating_5'], test_size=0.05)
SVM5 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM5.fit(Train_X, Train_Y)
predictions_SVM = SVM5.predict(Test_X)
score_5 = accuracy_score(predictions_SVM, Test_Y)*100
print("SVM Accuracy Score -> ",score_5)
#Get the importance for features
feature_importance = np.abs(SVM5.coef_)
sorted_indices = np.argsort(feature_importance)[0]
for i in reversed(sorted_indices):
    print(f"Feature {i}: Importance = {feature_importance[0][i]}")
#  df['region'], df['winery']], df['abv']
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
# predict the labels on validation dataset


# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", (score_1+score_2+score_3+score_4+score_5)/5)