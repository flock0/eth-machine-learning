import numpy as np
import time
import os
import sys
import nibabel as nib
from sklearn import preprocessing
from skimage.feature import hog
from skimage.feature import blob_dog, blob_log
from skimage.feature import canny
from skimage.util import view_as_windows
from skimage.morphology import disk
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif

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

def get_windowed_histogram(img_2d, WINDOW_SIZE = 10):
	STEP_SIZE = WINDOW_SIZE / 2
	NUM_BINS = 8
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
        feature.append(get_windowed_histogram(img_2d, 10))
    elif feature_id == 2:
        feature.append(get_windowed_histogram(img_2d, 20))
    elif feature_id == 3:
        feature.append(get_windowed_histogram(img_2d, 40))
    elif feature_id == 4:
        feature.append(get_windowed_histogram(img_2d, 80))
    elif feature_id == 5:
        feature.append(get_windowed_canny_histogram(img_2d, WINDOW_SIZE = 10))
    elif feature_id == 6:
        feature.append(get_windowed_canny_histogram(img_2d, WINDOW_SIZE = 20))  
    elif feature_id == 7:
        feature.append(get_windowed_canny_histogram(img_2d, WINDOW_SIZE = 40))
    elif feature_id == 8:
        feature.append(get_windowed_canny_histogram(img_2d, WINDOW_SIZE = 80))
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
    labels = np.append(np.loadtxt("./data/targets.csv", float), [-1]*TEST_SIZE)
    


    for feature_id in [1, 2, 3, 4, 5, 6, 7, 8]:
        OUTPUT_FILE = "{}/features_{}".format(OUTPUT_DIR, feature_id)
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
            for z in range(0, 176):
                img_2d = np.ascontiguousarray(img_3d[:,:,z][:,:,0])                    #(176, 208)
                # Features on every slice:
                if z % SLICE_INTERVAL == 0:
                    # Features on only a subset of slices:
                    feature.append(get_slice_feature(img_2d, feature_id))
            for x in range(0, 176):
                img_2d = np.ascontiguousarray(img_3d[x,:,:][:,:,0])                    #(176, 208)
                # Features on every slice:
                if x % SLICE_INTERVAL == 0:
                    # Features on only a subset of slices:
                    feature.append(get_slice_feature(img_2d, feature_id))
            for y in range(0, 208):
                img_2d = np.ascontiguousarray(img_3d[:,y,:][:,:,0])                    #(176, 176)
                # Features on every slice:
                if y % SLICE_INTERVAL == 0:
                    # Features on only a subset of slices:
                    feature.append(get_slice_feature(img_2d, feature_id))
            feature = np.hstack(feature)
            features.append(feature)

        features = np.array(features)
        print("Shape of the feature matrix:", features.shape)
        print("Standardization and reduction...")
        features_standardized = preprocessing.scale(features)
        pca = PCA()
        kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True)
        selection = SelectKBest(score_func=mutual_info_classif, k=TRAIN_SIZE).fit(features_standardized[:TRAIN_SIZE,:], labels[:TRAIN_SIZE])

        feature_pca = pca.fit_transform(features_standardized)
        feature_kpca = kpca.fit_transform(features_standardized)
        feature_kbest = selection.transform(features_standardized)
        features_reduced = np.concatenate((feature_pca, 
                                           feature_kpca,
                                           feature_kbest), axis=1)
        print("Shape of the reduced feature matrix:", features_reduced.shape)
        print("Saving to file ", OUTPUT_FILE)
        np.save(OUTPUT_FILE, features_reduced)
        print("Finished extraction!")

