#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.feature_selection import SelectKBest
from sklearn import preprocessing, naive_bayes, svm, linear_model
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees'] 
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
poi_label = ['poi']
features_list = poi_label + email_features + financial_features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
# Total number of data points
print("Total number of data points: %i" %len(data_dict))
# Allocation across classes (POI/non-POI)
poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
       poi += 1
print("Total number of poi: %i" % poi)
print("Total number of non-poi: %i" % (len(data_dict) - poi))
       
# Number of features used
all_features = data_dict[data_dict.keys()[0]].keys()
print("There are %i features for each person in the dataset, and %i features are used" %(len(all_features), len(features_list)))
# Are there features with many missing values? etc.
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict:
    for feature in data_dict[person]:
        if data_dict[person][feature] == "NaN":
            missing_values[feature] += 1
print("The number of missing values for each feature: ")
for feature in missing_values:
    print("%s: %i" %(feature, missing_values[feature]))
    
### Task 2: Remove outliers
def plotOutliers(data_set, feature_x, feature_y):
    data = featureFormat(data_set, [feature_x, feature_y])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter( x, y )
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()
# Visualize data to identify outliers
print(plotOutliers(data_dict, 'total_payments', 'total_stock_value'))
print(plotOutliers(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(plotOutliers(data_dict, 'salary', 'bonus'))

identity = []
for i in data_dict:
    if data_dict[i]['total_payments'] != "NaN":
        identity.append((i, data_dict[i]['total_payments']))
print("Outlier:")
print(sorted(identity, key = lambda x: x[1], reverse=True)[0:4])
# Remove outlier
data_dict.pop("TOTAL", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for person in my_dataset:
    msg_from_poi = my_dataset[person]['from_poi_to_this_person']
    to_msg = my_dataset[person]['to_messages']
    if msg_from_poi != "NaN" and to_msg != "NaN":
        my_dataset[person]['msg_from_poi_ratio'] = msg_from_poi/float(to_msg)
    else:
        my_dataset[person]['msg_from_poi_ratio'] = 0
    msg_to_poi = my_dataset[person]['from_this_person_to_poi']
    from_msg = my_dataset[person]['from_messages']
    if msg_to_poi != "NaN" and from_msg != "NaN":
        my_dataset[person]['msg_to_poi_ratio'] = msg_to_poi/float(from_msg)
    else:
        my_dataset[person]['msg_to_poi_ratio'] = 0
new_features_list = features_list + ['msg_to_poi_ratio', 'msg_from_poi_ratio']

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
selector = SelectKBest(k=12)
selector.fit(features, labels)
print("Best features:")
scores = zip(new_features_list,selector.scores_)
sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
optimized_features_list = poi_label + list(map(lambda x: x[0], sorted_scores))[0:12]
print(optimized_features_list)

new_data = featureFormat(my_dataset, optimized_features_list, sort_keys = True)
labels, features = targetFeatureSplit(new_data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
nb_clf = naive_bayes.GaussianNB()

k_clf = KMeans(n_clusters=2, tol=0.0001,random_state=42)

lo_clf = Pipeline(steps=[
        ('scaler', preprocessing.StandardScaler()),
        ('classifier', linear_model.LogisticRegression(tol = 0.0001, C = 10**-4, penalty = 'l2', random_state = 42))])

s_clf = svm.SVC(kernel='rbf', C=100, gamma = 0.001, random_state = 42, class_weight = 'balanced')
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"
def evaluate_clf(clf, features, labels, t=1000):
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
    print "accuracy: {}".format(accuracy)
    print "precision: {}".format(precision)
    print "recall:    {}".format(recall)

    return accuracy, precision, recall
print("Evaluate naive bayes model")
evaluate_clf(nb_clf, features, labels)
print("Evaluate k-mean model")
evaluate_clf(k_clf, features, labels)
print("Evaluate logistic regression model")
evaluate_clf(lo_clf, features, labels)
print("Evaluate svm model")
evaluate_clf(s_clf, features, labels)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(lo_clf, my_dataset, optimized_features_list)