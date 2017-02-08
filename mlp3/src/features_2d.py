import numpy as np
import time
import os
import sys
from util import *
import nibabel as nib
from sklearn import preprocessing
from skimage.feature import hog
from skimage.feature import blob_dog, blob_log
from skimage.feature import canny
from skimage.util import view_as_windows
from skimage.morphology import disk
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif

def get_blobs_log_histogram(img_2d):
	BINS = 10
	# The parameters for blog_log and histogram are (more or less) randomly chosen
	blobs_log = blob_log(img_2d, max_sigma=15, num_sigma=100, threshold=.01)
	if blobs_log.size != 0:
		histogram = np.histogram(blobs_log[:, 2], bins=BINS, density=True, range=(0, 25))
		feature = histogram[0] # Take the densities as the feature
	else:
		feature = np.zeros(BINS)
	return feature

def get_blobs_dog_histogram(img_2d):
	BINS = 6
	# The parameters for blog_dog and histogram are (more or less) randomly chosen
	blobs_dog = blob_dog(img_2d, min_sigma= 1, max_sigma=15, threshold=.01)
	if blobs_dog.size != 0:
		histogram = np.histogram(blobs_dog[:, 2], bins=BINS, density=True, range=(0, 15))
		feature = histogram[0] # Take the densities as the feature
	else:
		feature = np.zeros(BINS)
	return feature

def get_means(img_2d):
	return np.mean(img_2d)

def get_simple_flattened(img_3d):
    data = img_3d.flatten()
    #print(np.histogram(data, 10))
    return data

def get_histogram(img_2d):
	NUM_BINS = 8
	GLOBAL_MIN = 0
	GLOBAL_MAX = 4419
	return np.histogram(img_2d, bins=NUM_BINS, range=(GLOBAL_MIN, GLOBAL_MAX))[0]

def get_windowed_histogram(img_2d, WINDOW_SIZE = 10, NUM_BINS = 8):
	STEP_SIZE = WINDOW_SIZE / 2
	GLOBAL_MIN = 0
	GLOBAL_MAX = 4419
	all_windows = view_as_windows(img_2d, (WINDOW_SIZE, WINDOW_SIZE), STEP_SIZE)
	flattened_windows = np.reshape(all_windows, (-1, WINDOW_SIZE, WINDOW_SIZE))
	histogram_feature_list = []
	i = 1
	for window in flattened_windows:
		single_hist = np.histogram(window, bins=NUM_BINS, range=(GLOBAL_MIN, GLOBAL_MAX))[0]
		histogram_feature_list.append(single_hist)
	return np.hstack(histogram_feature_list)

def get_windowed_canny_histogram(img_2d, WINDOW_SIZE = 10):
	"""Whereas previously we worked with a histogram, now we will average over window
	Function takes an image as an input and returns """
	feature_list = []
	for threshold in [1,3,5]:
		STEP_SIZE = WINDOW_SIZE / 2
		NUM_BINS = 2
		all_windows = view_as_windows(img_2d, (WINDOW_SIZE, WINDOW_SIZE), STEP_SIZE)
		flattened_windows = np.reshape(all_windows, (-1, WINDOW_SIZE, WINDOW_SIZE))
		for window in flattened_windows:
			m = np.histogram(canny(window, threshold, 1, 25).flatten(), bins=NUM_BINS)[0]
			feature_list.append(m)
	return np.hstack(feature_list)

def get_slice_feature(img_2d, feature_id): # for greyscale image
    '''make sure everything you append to it is a 1 dimension vector'''
    feature = []
    if feature_id == 1:
        feature.append(get_windowed_canny_histogram(img_2d, WINDOW_SIZE =10))
    elif feature_id == 2:
        feature.append(get_windowed_canny_histogram(img_2d, WINDOW_SIZE = 20))
    elif feature_id == 3:
        feature.append(get_windowed_canny_histogram(img_2d, WINDOW_SIZE = 40))
    else:
        print("Error in get_slice_feature(): unknown features id.")
    return np.hstack(feature)

if __name__=='__main__':

    DATA_DIR = "./data"
    OUTPUT_DIR = "./features"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    TRAIN_SIZE = 278
    TEST_SIZE = 138
    SLICE_INTERVAL = 30

    TRAIN_DIR = DATA_DIR + "/set_train"
    TEST_DIR = DATA_DIR + "/set_test"
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    labels = read_label_multi_label('./data/targets.csv')

    for feature_id in [2, 3]:
        print("Vanilla 2d features:", feature_id, TIMESTAMP)
        # naming data files appropirately
        if feature_id == 1:
            OUTPUT_FILE = "{}/features_2d_canny_THRES123_SLICEINT{}_WINDOW10".format(OUTPUT_DIR, SLICE_INTERVAL)
        elif feature_id == 2:
            OUTPUT_FILE = "{}/features_2d_canny_THRES123_SLICEINT{}_WINDOW20".format(OUTPUT_DIR, SLICE_INTERVAL)
        elif feature_id == 3:
            OUTPUT_FILE = "{}/features_2d_canny_THRES123_SLICEINT{}_WINDOW40".format(OUTPUT_DIR, SLICE_INTERVAL)
        else:
            print("Delta tango...")

        features = []

        print("Extracting 2D features...")
        print(OUTPUT_FILE, TIMESTAMP)
        inputFileName = "train"
        inputFileDirectory = TRAIN_DIR
        j = 0
        for i in range(1, TRAIN_SIZE + TEST_SIZE + 1):
            j += 1
            if i == TRAIN_SIZE + 1:
                inputFileName = "test"
                inputFileDirectory = TEST_DIR
                j = 1
            nii_file_path = "{}/{}_{}.nii".format(inputFileDirectory, inputFileName, j)
            nii_file = nib.load(nii_file_path)
            print("Extracting 2D features from {}_{}".format(inputFileName, j))
            img_3d = np.array(nii_file.get_data())             #(176, 208, 176)
            feature = []
            # Iterate through all slices
            for z in range(30, 146): # range reduced, orig 0 176
                img_2d = np.ascontiguousarray(img_3d[:,:,z][:,:,0])                    #(176, 208)
                # Features on every slice:
                if z % SLICE_INTERVAL == 0:
                    # Features on only a subset of slices:
                    feature.append(get_slice_feature(img_2d, feature_id))
            for x in range(30, 146): # range reduced, orig 0 176
                img_2d = np.ascontiguousarray(img_3d[x,:,:][:,:,0])                    #(176, 208)
                # Features on every slice:
                if x % SLICE_INTERVAL == 0:
                    # Features on only a subset of slices:
                    feature.append(get_slice_feature(img_2d, feature_id))
            for y in range(30, 178): # range reduced, orig 0 208
                img_2d = np.ascontiguousarray(img_3d[:,y,:][:,:,0])                    #(176, 176)
                # Features on every slice:
                if y % SLICE_INTERVAL == 0:
                    # Features on only a subset of slices:
                    feature.append(get_slice_feature(img_2d, feature_id))
            feature = np.hstack(feature)
            features.append(feature)

        features = np.array(features)
        print("Shape of the feature matrix:", features.shape)
        print("Removing features which do not vary...")
        features = features[:, np.std(features, axis=0) != 0]
        print("Shape of the feature matrix:", features.shape)
        print("Removing features with NAs...")
        features = np.delete(features, np.where(sum(np.isnan(features)) != 0), 1)
        print features.shape
        print("Removing features which are not finite...")
        features = np.delete(features,np.where(sum(np.isfinite(features)) == 0), 1)
        print features.shape
        print("Standardization and reduction...")
        features_standardized = preprocessing.scale(features)
        pca = PCA()
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
        skb = SelectKBest(score_func=f_classif, k=TRAIN_SIZE)
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
        np.save(OUTPUT_FILE, preprocessing.scale(features_reduced))
        print("Finished extraction!")
