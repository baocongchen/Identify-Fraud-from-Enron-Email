#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here--now create 2 decision tree classifiers,
### one with min_samples_split=2 and one with min_samples_split=50
### compute the accuracies on the testing data and store
### the accuracy numbers to acc_min_samples_split_2 and
### acc_min_samples_split_50, respectively
#clf2 = tree.DecisionTreeClassifier(min_samples_split=2)
clf40 = tree.DecisionTreeClassifier(min_samples_split=40)
#clf50 = tree.DecisionTreeClassifier(min_samples_split=50)
#clf2 = clf2.fit(features_train, labels_train)
clf40 = clf40.fit(features_train, labels_train)
#clf50 = clf50.fit(features_train, labels_train)
#acc_min_samples_split_2 = clf2.score(features_test, labels_test)
#acc_min_samples_split_50 = clf50.score(features_test, labels_test)
acc_min_samples_split_40 = clf40.score(features_test, labels_test)
print("accuracy when min_samples_split is 40: %f" %acc_min_samples_split_40)
#print(len(features_train[0]))
#########################################################


