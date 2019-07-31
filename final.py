#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:25:07 2019

@author: huasongzhang
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import tree


train = pd.read_csv('train_low.csv')
(a,b) = train.shape
    
test = pd.read_csv('test_low.csv')
(a1,b1) = test.shape

y_train = np.array([train.category_id])
x_train = np.array([train.views,train.likes,train.dislikes,train.comment_count,
train.diff_days,train.like_percentage])

y_test = np.array([test.category_id])
x_test = np.array([test.views,test.likes,test.dislikes,test.comment_count,
test.diff_days,test.like_percentage])

# plot the frequency appearances for each category(low)
categories_freq = train['category_id'].value_counts()
categories_freq.sort_index(inplace=True)
frequency = categories_freq.values
categories = categories_freq.index.values
plt.bar(list(map(str,categories)), frequency)
plt.xlabel('Category id')
plt.ylabel('Count')
plt.title('Trending Videos Category(low)')
plt.show()

# plot the like percentage for each category
train_like_per = train[['category_id', 'like_percentage']].copy()
like_per = train_like_per.groupby(['category_id']).mean()
plt.bar(list(map(str,categories)), like_per.like_percentage.values)
plt.xlabel('Category id')
plt.ylabel('Likes Percentage')
plt.title('Percentage of Likes Category(low)')
plt.show()

# plot Trending Videos by Channel(low)
channel_freq = train['channel_title'].value_counts()
low_channel_freq = channel_freq[0:10]
low_channel_freq.sort_index(inplace=True)
frequency_chan = low_channel_freq.values
channels = low_channel_freq.index.values
plt.bar(list(map(str,channels))[0:10], frequency_chan[0:10])
plt.xticks(rotation='vertical')
plt.xlabel('Channel')
plt.ylabel('Count')
plt.title('Trending Videos by Channel(low)')
plt.show()

def accuracy(truth,prediction):
    n = len(truth)
    count = 0
    for i in range(n):
        if truth[i] == prediction[i]:
            count += 1
    return count/n

# knn
nn = np.array([i for i in range(1,51)])
Acc_knn = []
for i in range(len(nn)):
    a = nn[i]
    classifier = KNeighborsClassifier(n_neighbors=a)  
    fit_knn = classifier.fit(np.transpose(x_train), np.ravel(np.transpose(y_train)))
    y_pred_knn = fit_knn.predict(np.transpose(x_test))
    acc_knn = accuracy(np.ravel(np.transpose(y_test)),y_pred_knn)
    Acc_knn.append(acc_knn)
max_acc_knn = max(Acc_knn)
plt.plot(nn,Acc_knn)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.title('Accuracy for KNN')
plt.show()


# decision tree
classifier1 = tree.DecisionTreeClassifier()
fit_tree = classifier1.fit(np.transpose(x_train), np.ravel(np.transpose(y_train)))
y_pred_tree = fit_tree.predict(np.transpose(x_test))
acc_tree = accuracy(np.ravel(np.transpose(y_test)),y_pred_tree)


        











    
    
    