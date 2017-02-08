import os
import sys
import numpy as np
import nibabel as nib
import cv2

from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from scipy.stats import uniform, randint
from myutil import *

def s_error(scores, best_score):
    for i in range(len(scores)):
        if(best_score == scores[i][1]):
            return np.sqrt(np.var(scores[i][2]))

if __name__=='__main__':
    print("3_model_selection_gridsearch")
    TRAIN_SIZE = 278
    TEST_SIZE = 0
    FEATURES_FILE = sys.argv[1]
    print("Reading labels...")
    labels = np.append(np.loadtxt("./targets.csv", int), [-1]*TEST_SIZE)
    print("Reading feature matrix...")
    features = np.load(FEATURES_FILE)
    print("Shape of the feature matrix",features.shape)

    CROSS_VALIDATION_STEPS = 4
    N_JOBS = 4

    ### Model selection and parameter tuning ###
    print("Model Selection...")
    rr = linear_model.Ridge()
    lasso = linear_model.Lasso()
    linreg = linear_model.LinearRegression()
    svr = SVR(kernel='rbf')

    # Setup pipeline
    union = FeatureUnion([('kernelpca', KernelPCA(kernel='rbf', n_jobs=N_JOBS)), ('kbest', SelectKBest(f_regression))], n_jobs=N_JOBS)
    pipeRR = Pipeline([('featureunion', union), ('rr', rr)])
    pipeLasso = Pipeline([('featureunion', union), ('lasso', lasso)])
    pipeLinReg = Pipeline([('featureunion', union), ('linreg', linreg)])
    pipeSVR = Pipeline([('featureunion', union), ('svr', svr)])
    ### Hyperparameter tuning for various regression models using GridSearchCV ###
    best_scores = []

    # Search space for the parameter tuning
    kernelPCAKernel = ['linear', 'rbf', 'poly']
    kBest = np.array([500, 1000, 10000])
    rrAlphas = np.array([10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0])
    lassoAlphas = np.array([10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001])
    lassoTols = np.array([1, 0.1, 0.01, 0.05])
    svrCs = np.array([1e0, 1e1, 1e2, 1e3])
    svrGammas = np.logspace(-5, 2, 5)

    # Find the optimal parameter for Linear regression
    print("Linear regression...")
    grid = GridSearchCV(estimator=pipeLinReg, param_grid=dict(featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__kbest__k=kBest), scoring="mean_squared_error", cv=CROSS_VALIDATION_STEPS, n_jobs=N_JOBS)
    grid.fit(features[:TRAIN_SIZE], labels[:TRAIN_SIZE])
    best_scores.append(["LinReg", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_)
, grid.best_params_])

    # Find the optimal parameter for Ridge regression
    print("Ridge regression...")
    grid = GridSearchCV(estimator=pipeRR, param_grid=dict(rr__alpha=rrAlphas, featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__kbest__k=kBest), scoring="mean_squared_error", cv=CROSS_VALIDATION_STEPS, n_jobs=N_JOBS)
    grid.fit(features[:TRAIN_SIZE], labels[:TRAIN_SIZE])
    best_scores.append(["Ridge", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_)
, grid.best_params_])

    # Find the optimal parameter for LASSO
    print("Lasso...")
    grid = GridSearchCV(estimator=pipeLasso, param_grid=dict(lasso__alpha=lassoAlphas, lasso__tol=lassoTols, featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__kbest__k=kBest), scoring="mean_squared_error", cv=CROSS_VALIDATION_STEPS, n_jobs=N_JOBS)
    grid.fit(features[:TRAIN_SIZE], labels[:TRAIN_SIZE])
    best_scores.append(["LASSO", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_)
, grid.best_params_])

    # Find the optimal parameter for SVR
    print("SVR...")
    grid = GridSearchCV(estimator=pipeSVR, param_grid=dict(svr__C=svrCs, svr__gamma=svrGammas, featureunion__kernelpca__kernel=kernelPCAKernel, featureunion__kbest__k=kBest), scoring="mean_squared_error", cv=CROSS_VALIDATION_STEPS, n_jobs=N_JOBS)
    grid.fit(features[:TRAIN_SIZE], labels[:TRAIN_SIZE])
    best_scores.append(["SVR", abs(grid.best_score_), s_error(grid.grid_scores_, grid.best_score_)
, grid.best_params_])

    # Overview of all models
    bestScoresArray = np.array(best_scores)
    print best_scores
    print("Best model with optimal parameter", bestScoresArray[np.argmin(bestScoresArray[:, 1])])