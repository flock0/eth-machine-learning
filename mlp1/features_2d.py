import numpy as np
from skimage.feature import hog
from skimage.feature import blob_dog, blob_log
from skimage.util import view_as_windows
from skimage.morphology import disk

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

def get_slice_feature(img_2d): # for greyscale image
    '''make sure everything you append to it is a 1 dimension vector'''
    feature = []
    #feature.append(hog(img_2d, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualise=False, normalise=None).ravel())
    feature.append(hog(img_2d, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(3, 3), visualise=False, normalise=None).ravel())
    #feature.append(get_blobs_log_histogram(img_2d))
    #feature.append(get_blobs_dog_histogram(img_2d))
    #feature.append(get_windowed_histogram(img_2d, 10))
    #feature.append(get_windowed_histogram(img_2d, 20))
    #feature.append(get_windowed_histogram(img_2d, 40))
    #feature.append(get_windowed_histogram(img_2d, 80))

    # TODO haar feature
    return np.hstack(feature)
