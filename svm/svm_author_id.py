#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]                                 
print("Ready")
#==============================================================================
# for i in range(1,5):
#     clf = svm.SVC(kernel="rbf", C=1.0*10**i)
#     t0 = time()
#     clf = clf.fit(features_train, labels_train)
#     t1 = round(time() - t0, 3)
#     print("Training time: %f" % t1)
#     t2 = time()
#     pred = clf.predict(features_test)
#     t3 = round(time() - t2, 3)
#     print("Prediction time: %f" % t3)
#     acc = clf.score(features_test, labels_test)
#     print "Accuracy: ",acc
#==============================================================================
clf = svm.SVC(kernel="rbf", C=10000)
t0 = time()
clf = clf.fit(features_train, labels_train)
t1 = round(time() - t0, 3)
print("Training time: %f" % t1)
t2 = time()
pred = clf.predict(features_test)
print("Predicted value: %f" % pred[50])
print("Number of observations predicted to be Chris: %f" % sum(pred))
t3 = round(time() - t2, 3)
print("Prediction time: %f" % t3)
acc = clf.score(features_test, labels_test)
print "Accuracy: ",acc

#########################################################


