
import numpy as np
import os
import cv2



''' 
just a bunch of different functions with different tasks.



'''


def resize(image, size):
	return cv2.resize(image, size)


def flattenImage(image):
	n_rows, n_cols, _ = image.shape
	return image.reshape((n_rows*n_cols), 3)


'''
apply function sequentially/serially to the data
'''
def featurize(fns, data):
	for f in fns:
		out_data = map(data, f)

	return out_data



