import os
import time
import sys
import numpy as np
import nibabel as nib
import cv2
from sklearn import preprocessing

from myutil import *
from features_2d import *
from features_3d import *

def get_features(img_3d, SLICE_INTERVAL = 30):
    feature = []
    # Iterate through all slices
    for z in range(0, 176):
        img_2d = np.ascontiguousarray(img_3d[:,:,z][:,:,0])                    #(176, 208)
        # Features on every slice:
        if z % SLICE_INTERVAL == 0:
            # Features on only a subset of slices:
            feature.append(get_slice_feature(img_2d))
    for x in range(0, 176):
        img_2d = np.ascontiguousarray(img_3d[x,:,:][:,:,0])                    #(176, 208)
        # Features on every slice:
        if x % SLICE_INTERVAL == 0:
            # Features on only a subset of slices:
            feature.append(get_slice_feature(img_2d))
    for y in range(0, 208):
        img_2d = np.ascontiguousarray(img_3d[:,y,:][:,:,0])                    #(176, 176)
        # Features on every slice:
        if y % SLICE_INTERVAL == 0:
            # Features on only a subset of slices:
            feature.append(get_slice_feature(img_2d))
    return np.hstack(feature)

print("1_extract_features")
TRAIN_SIZE = 278
TEST_SIZE = 138
TRAIN_DIR = "./data/set_train"
TEST_DIR = "./data/set_test"
OUTPUT_DIR = "./features"
FEATURESET_NAME = "all"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = "./features"
features = []

print("Extracting features...")
SLICE_INTERVAL = 30
for i in range(1, TRAIN_SIZE + 1):
    nii_file_path = "{}/train_{}.nii".format(TRAIN_DIR, i)
    nii_file = nib.load(nii_file_path)         
    print(nii_file_path)          
    img_3d = np.array(nii_file.get_data())             #(176, 208, 176)
    feature = get_features(img_3d, SLICE_INTERVAL)
    features.append(feature)
for i in range(1, TEST_SIZE + 1):
    nii_file_path = "{}/test_{}.nii".format(TEST_DIR, i)
    nii_file = nib.load(nii_file_path)               
    print(nii_file_path)     
    img_3d = np.array(nii_file.get_data())             #(176, 208, 176)
    feature = get_features(img_3d, SLICE_INTERVAL)
    features.append(feature)
features = np.array(features)
windowed_hist_3d = np.array(get_windowed_histogram_3d(TRAIN_DIR, TEST_DIR, TRAIN_SIZE, TEST_SIZE, NUM_BINS = 8))
features = np.append(features, windowed_hist_3d, axis=1)
print("Shape of the feature matrix:", features.shape)
print("Standardization...")
features_normalized = preprocessing.scale(features[:])
print("Saving to file ", OUTPUT_FILE)
np.save(OUTPUT_FILE, features_normalized)
print("Finished extraction!")