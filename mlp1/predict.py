import os
import sys
import numpy as np
import nibabel as nib
import cv2

from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from scipy.stats import uniform, randint
from myutil import *

print("7_predict")
TRAIN_SIZE = 278
TEST_SIZE = 138
FEATURES_FILE = "./features.npy"
N_JOBS = 8
print("Reading labels...")
labels = np.append(np.loadtxt("./targets.csv", int), [-1]*TEST_SIZE)
print("Reading feature matrix...")
features = np.load(FEATURES_FILE)
print("Shape of the feature matrix",features.shape)

print("Predicting...") # Plug in the optimal model from previous steps
union = FeatureUnion([('kernelpca', KernelPCA(kernel='poly')), ('kbest', SelectKBest(f_regression, k = 25000))])
model = Pipeline([('featureunion', union), ('svr', SVR(kernel='rbf',C=1000.0, gamma=1e-5))])

model.fit(features[:TRAIN_SIZE], labels[:TRAIN_SIZE])
predicted = model.predict(features[TRAIN_SIZE:])
write_submission(predicted)
