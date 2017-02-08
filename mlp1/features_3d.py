import os
import time
import sys
import numpy as np
import nibabel as nib
import cv2
from sklearn import preprocessing
from skimage.util import view_as_windows
from myutil import *

def get_windowed_histogram_3d(TRAIN_DIR, TEST_DIR, TRAIN_SIZE, TEST_SIZE, NUM_BINS = 8):
    features = []

    print("Extracting 3D sliding window features...")
    #print("Initializing sliding windows...")
    # Initialize and gather number of sliding windows
    IMAGE_DIMENSIONS = (176, 208, 176)
    WINDOW_SIZE = 16
    STEP_SIZE = 8
    zeroimg = np.zeros(IMAGE_DIMENSIONS)
    all_windows = view_as_windows(zeroimg, WINDOW_SIZE, STEP_SIZE)
    flat_wins = all_windows.reshape((-1, WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE))
    totalNumberWindows = flat_wins.shape[0]
    minIntensities = np.full(totalNumberWindows, sys.maxsize)
    maxIntensities = np.full(totalNumberWindows, -sys.maxsize-1)

    #print("Gathering min max values...")
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
        #print "Gathering min max from {}_{}".format(inputFileName, j)
        nii_file_path = "{}/{}_{}.nii".format(inputFileDirectory, inputFileName, j)
        nii_file = nib.load(nii_file_path)         
        img_4d = np.array(nii_file.get_data())             #(176, 208, 176)
        img_3d = np.ascontiguousarray(img_4d.reshape(IMAGE_DIMENSIONS))
        windows = view_as_windows(img_3d, WINDOW_SIZE, STEP_SIZE)
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
        #print "Extracting from {}_{}".format(inputFileName, j)
        nii_file_path = "{}/{}_{}.nii".format(inputFileDirectory, inputFileName, j)
        nii_file = nib.load(nii_file_path)         
        img_4d = np.array(nii_file.get_data())             #(176, 208, 176)
        img_3d = np.ascontiguousarray(img_4d.reshape(IMAGE_DIMENSIONS))
        windows = view_as_windows(img_3d, WINDOW_SIZE, STEP_SIZE)
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
