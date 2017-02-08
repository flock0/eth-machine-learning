import os
import time
import sys
import numpy as np
from util import *
import nibabel as nib
import cv2
from sklearn import preprocessing
from skimage.util import view_as_windows
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from util import *

def get_windowed_histogram_3d(TRAIN_DIR, TEST_DIR, TRAIN_SIZE, TEST_SIZE, WINDOW_SIZE = 16, STEP_SIZE = 8, NUM_BINS = 30):
    features = []

    print("Extracting 3D sliding window features...")

    # Initialize and gather number of sliding windows
    IMAGE_DIMENSIONS = (176, 208, 176)

    zeroimg = np.zeros(IMAGE_DIMENSIONS)
    all_windows = view_as_windows(zeroimg, (WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE), STEP_SIZE)
    flat_wins = all_windows.reshape((-1, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE))
    totalNumberWindows = flat_wins.shape[0]
    minIntensities = np.full(totalNumberWindows, sys.maxsize)
    maxIntensities = np.full(totalNumberWindows, -sys.maxsize-1)

    print("Gathering min max values...")
    # Gather min max values for each of the sliding windows
    inputFileName = "train"
    inputFileDirectory = TRAIN_DIR
    j = 0
    for i in range(1, TRAIN_SIZE + TEST_SIZE + 1):
        j += 1
        if i == TRAIN_SIZE + 1:
            inputFileName = "test"
            inputFileDirectory = TEST_DIR
            j = 1
        print("\t\tGathering min max from {}_{}".format(inputFileName, j))
        nii_file_path = "{}/{}_{}.nii".format(inputFileDirectory, inputFileName, j)
        nii_file = nib.load(nii_file_path)
        img_4d = np.array(nii_file.get_data())             #(176, 208, 176)
        img_3d = np.ascontiguousarray(img_4d.reshape(IMAGE_DIMENSIONS))
        windows = view_as_windows(img_3d, (WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE), STEP_SIZE)
        flattened_windows = windows.reshape((-1, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE))
        for w in range(0, totalNumberWindows):
            localMinIntensity = flattened_windows[w, :].min()
            if localMinIntensity < minIntensities[w]:
                minIntensities[w] = localMinIntensity
            localMaxIntensity = flattened_windows[w, :].max()
            if localMaxIntensity > maxIntensities[w]:
                maxIntensities[w] = localMaxIntensity


    print("\tExtracting histograms from sliding windows...")
    inputFileName = "train"
    inputFileDirectory = TRAIN_DIR
    j = 0
    for i in range(1, TRAIN_SIZE + TEST_SIZE + 1):
        j += 1
        if i == TRAIN_SIZE + 1:
            inputFileName = "test"
            inputFileDirectory = TEST_DIR
            j = 1
        print("\t\tExtracting from {}_{}".format(inputFileName, j))
        nii_file_path = "{}/{}_{}.nii".format(inputFileDirectory, inputFileName, j)
        nii_file = nib.load(nii_file_path)
        img_4d = np.array(nii_file.get_data())
        img_3d = np.ascontiguousarray(img_4d.reshape(IMAGE_DIMENSIONS))
        img_3d = img_3d[30:146, 30:178, 30:146] #reduced from (176, 208, 176)
        windows = view_as_windows(img_3d, (WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE), STEP_SIZE)
        flattened_windows = windows.reshape((-1, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE))
        histogram_feature_list = []
        for w in range(0, flattened_windows.shape[0]):
            window = flattened_windows[w, :]
            single_hist = np.histogram(window, bins=NUM_BINS, density=True, range=(minIntensities[w], maxIntensities[w]))[0]
            histogram_feature_list.append(single_hist)
        features.append(np.hstack(histogram_feature_list))

    features = np.array(features)
    print("\tshape:", features.shape)
    return features

if __name__=='__main__':
    DATA_DIR = "./data"
    OUTPUT_DIR = "./features"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    FEATURESET_NAME = ""
    TRAIN_SIZE = 278
    TEST_SIZE = 138
    SLICE_INTERVAL = 30

    TRAIN_DIR = DATA_DIR + "/set_train"
    TEST_DIR = DATA_DIR + "/set_test"
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    labels = read_label_multi_label('./data/targets.csv')
    for feature_id in range(3):
        OUTPUT_FILE = "3d features".format(OUTPUT_DIR, feature_id)
        print(OUTPUT_FILE, TIMESTAMP)
        if feature_id == 0:
            OUTPUT_FILE = "{}/features_3d_WINDOW_SIZE8STEP_SIZE4NUM_BINS30".format(OUTPUT_DIR)
            features = np.array(get_windowed_histogram_3d(TRAIN_DIR, TEST_DIR, TRAIN_SIZE, TEST_SIZE, WINDOW_SIZE = 8, STEP_SIZE = 4, NUM_BINS = 30))
        elif feature_id == 1:
            OUTPUT_FILE = "{}/features_3d_WINDOW_SIZE16STEP_SIZE8NUM_BINS30".format(OUTPUT_DIR)
            features = np.array(get_windowed_histogram_3d(TRAIN_DIR, TEST_DIR, TRAIN_SIZE, TEST_SIZE, WINDOW_SIZE = 16, STEP_SIZE = 8, NUM_BINS = 30))
        elif feature_id == 2:
            OUTPUT_FILE = "{}/features_3d_WINDOW_SIZE32STEP_SIZE16NUM_BINS30".format(OUTPUT_DIR)
            features = np.array(get_windowed_histogram_3d(TRAIN_DIR, TEST_DIR, TRAIN_SIZE, TEST_SIZE, WINDOW_SIZE = 32, STEP_SIZE = 16, NUM_BINS = 30))
        print("Shape of the feature matrix:", features.shape)
        print("Removing features which do not vary...")
        features = features[:, np.std(features, axis=0) != 0]
        print("Shape of the feature matrix:", features.shape)
        print("Removing features with NAs...")
        features = np.delete(features, np.where(sum(np.isnan(features)) != 0), 1)
        print features.shape

        print("Removing features which are not finite...")
        features = np.delete(features,np.where(sum(np.isfinite(features)) == 0),1)
        print features.shape
        print("Standardization and reduction...")
        features_standardized = preprocessing.scale(features)
        pca = PCA()
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
        skb = SelectKBest(score_func=f_classif, k=TRAIN_SIZE)
        # since for each class the same feature vector might have important information
        # need to make sure that we are not apending duplicate columns to our final master
        # feature vector. Hence finding the union and subseting our the columns which
        # are contained in the union of the features selected
        print("Finding the union of select k best feature vectors for 3 regressors...")
        selection0_id = skb.fit(features_standardized[:TRAIN_SIZE,:], labels[:TRAIN_SIZE,0]).get_support(indices = True)
        selection1_id = skb.fit(features_standardized[:TRAIN_SIZE,:], labels[:TRAIN_SIZE,1]).get_support(indices = True)
        selection2_id = skb.fit(features_standardized[:TRAIN_SIZE,:], labels[:TRAIN_SIZE,2]).get_support(indices = True)
        selection_union = np.union1d(np.union1d(selection0_id, selection1_id), selection2_id)

        feature_pca = pca.fit_transform(features_standardized)
        print("Shape of PCA features: {}".format(feature_pca.shape))
        feature_kpca = kpca.fit_transform(features_standardized)
        print("Shape of Kernel PCA features: {}".format(feature_kpca.shape))
        feature_kbest_union = features_standardized[:,selection_union] # columns selected by the SelectKBest algo
        print("Shape of Select K Best features: {}".format(feature_kbest_union.shape))
        features_reduced = np.concatenate((feature_pca,
		                                   feature_kpca,
		                                   feature_kbest_union
                                            ),
                                           axis=1)
        print("Shape of the reduced feature matrix:", features_reduced.shape)
        print("Saving to file ", OUTPUT_FILE)
        np.save(OUTPUT_FILE, features_reduced)
        print("Finished extraction!")
