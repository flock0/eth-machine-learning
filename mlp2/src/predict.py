import os
import sys
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, log_loss
from myutil import *

TRAIN_SIZE = 278
TEST_SIZE = 138
TRAIN_DIR = "./data/set_train"
TEST_DIR = "./data/set_test"
labels = np.append(np.loadtxt("./data/targets.csv", float), [-1]*TEST_SIZE)
aggregate_features = np.ones((416, 1))
for feature_id in [1,2,3,4,5,6,7,8,9,10,11]:
    filename = "./features/features_{}.npy".format(feature_id)
    features = np.load(filename)
    print("shape of {}:".format(filename) , features.shape)
    aggregate_features = np.append(aggregate_features, features, axis=1)
features_standardized = preprocessing.scale(aggregate_features)
#selection = SelectKBest(score_func=mutual_info_classif, k=1000).fit(features_standardized[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])
#features_reduced = selection.transform(features_standardized)
features_reduced = features_standardized
print("shape of reduced features:", features_reduced.shape)

clf = RandomForestClassifier(n_estimators=280, criterion='entropy', class_weight={1:67,0:221})
#clf = RandomForestClassifier(n_estimators=280, criterion='gini', class_weight={1:67,0:221})
clf.fit(features_reduced[:TRAIN_SIZE], labels[:TRAIN_SIZE])
print(log_loss(labels[:TRAIN_SIZE], clf.predict_proba(features_reduced[:TRAIN_SIZE])))
print(confusion_matrix(labels[:TRAIN_SIZE], clf.predict(features_reduced[:TRAIN_SIZE])))
predicted = clf.predict_proba(features_reduced[TRAIN_SIZE:])
write_submission(predicted[:,1])
