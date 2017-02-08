import os
import sys
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn import ensemble
from sklearn.metrics import f1_score, hamming_loss, make_scorer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import KernelPCA
from util import *

TRAIN_SIZE = 278
TEST_SIZE = 138
TRAIN_DIR = "./data/set_train"
TEST_DIR = "./data/set_test"
FEATURE_FILE = "./features_combined.npy"
labels = np.append(read_label_multi_label('./data/targets.csv'), np.full((TEST_SIZE, 3), -1.0), axis=0)
aggregate_features = np.load(FEATURE_FILE)
features_reduced = preprocessing.scale(aggregate_features)


w = [{1:108,0:170}, {1:133,0:145}, {1:67,0:211}]

label_type = {0:"gender",
              1:"age",
              2:"health"}

svm = SVC(C=0.1, gamma=1.0, kernel='linear', class_weight=w[0])
union = FeatureUnion([('kernelpca', KernelPCA(kernel='rbf')), ('perc', SelectPercentile(f_classif, percentile=15))])
clf = Pipeline([('featureunion', union), ('svm', svm)])
clf.fit(features_reduced[:TRAIN_SIZE], labels[:TRAIN_SIZE, 0])
print(confusion_matrix(labels[:TRAIN_SIZE, 0], clf.predict(features_reduced[:TRAIN_SIZE])))
labels[TRAIN_SIZE:, 0] = clf.predict(features_reduced[TRAIN_SIZE:])


svm = SVC(C=0.1, gamma=0.0001, kernel='rbf', class_weight=w[1])
union = FeatureUnion([('kernelpca', KernelPCA(kernel='rbf')), ('perc', SelectPercentile(f_classif, percentile=15))])
clf = Pipeline([('featureunion', union), ('svm', svm)])
clf.fit(features_reduced[:TRAIN_SIZE], labels[:TRAIN_SIZE, 1])
print(confusion_matrix(labels[:TRAIN_SIZE, 1], clf.predict(features_reduced[:TRAIN_SIZE])))
labels[TRAIN_SIZE:, 1] = clf.predict(features_reduced[TRAIN_SIZE:])

svm = SVC(C=0.1, gamma=0.1, kernel='linear', class_weight=w[2])
union = FeatureUnion([('kernelpca', KernelPCA(kernel='rbf')), ('perc', SelectPercentile(f_classif, percentile=18))])
clf = Pipeline([('featureunion', union), ('svm', svm)])
clf.fit(features_reduced[:TRAIN_SIZE], labels[:TRAIN_SIZE, 2])
print(confusion_matrix(labels[:TRAIN_SIZE, 2], clf.predict(features_reduced[:TRAIN_SIZE])))
labels[TRAIN_SIZE:, 2] = clf.predict(features_reduced[TRAIN_SIZE:])
predicted_labels = labels[TRAIN_SIZE:]

write_submission_multi_label(labels[TRAIN_SIZE:])
