import pandas as pd
import numpy as np
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score

df = pd.read_csv('WineData.csv')
Train_X, Test_X, Train_Y, Test_Y = [], [], [], []
count = 0
for index, row in df.iterrows():
    if count > df.__len__() * 0.05:
        Train_X.append([df['type'], df['ela'], df['grapes'], df['harmon'], df['body'], df['acid'],
                        df['code'], df['region'], df['winery'], df['abv']])
        Train_Y.append(df['rating'])
    else:
        Test_X.append([df['type'], df['ela'], df['grapes'], df['harmon'], df['body'], df['acid'],
                       df['code'], df['region'], df['winery'], df['abv']])
        Test_Y.append(df['rating'])

    count += 1

"""
model_selection.train_test_split(
    [df['type'], df['ela'], df['grapes'], df['harmon'], df['body'], df['acid'],
     df['code']],
    df['rating'],
    test_size=0.05
)
"""
#  df['region'], df['winery']], df['abv']
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM1 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM1.fit(Train_X, Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM1.predict(Test_X)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)