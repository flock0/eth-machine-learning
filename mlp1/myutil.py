import os
import time
import numpy as np
import nibabel as nib
import cv2

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use('Agg') # for server w/o display
import matplotlib.pyplot as plt

def real_train_test(X, y, train_size, clf):
    clf.fit(X[:train_size], y[:train_size], get_weights(y[:train_size], n_bin = 10))
    return clf.predict(X[train_size:])

def write_submission(y):
    with open("submission.csv", "w") as fw:
        fw.write("ID,Prediction\n")
        i = 1
        for yi in y:
            fw.write("{},{}\n".format(i, yi))
            i += 1
    
def plot_pca_data(pca, features_new, labels):
    for i in range(3):
        fig, ax = plt.subplots()
        ax.scatter(features_new[:,i], labels[:])
        ax.plot([features_new.min(), features_new.max()], [labels.min(), labels.max()], ls="--", lw=4)
        ax.set_xlabel('pc{}'.format(i))
        ax.set_ylabel('label')
        plt.savefig("./pc{}_cv10.png".format(i))
    fig, ax = plt.subplots()
    ax.plot(pca.explained_variance_, linewidth=2)
    plt.savefig("./explained_variance.png".format(i))

def plot_cv_prediction(labels, predicted):
    fig, ax = plt.subplots()
    ax.scatter(labels, predicted)
    ax.plot([labels.min(), labels.max()], [predicted.min(), predicted.max()], ls="--", lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig("./regression_cv10.png")

def get_weights(train_label, n_bin = 10):
    hist = np.histogram(train_label, bins=n_bin)
    counts = hist[0]
    bins = hist[1]
    n_label = sum(bins)
    weights = []
    for l in train_label:
        bin_id = next(b[0] for b in enumerate(bins) if b[1] >= l) - 1
        #print(l, bins[bin_id]/n_label)
        weights.append(bins[bin_id]/n_label)
    return weights

