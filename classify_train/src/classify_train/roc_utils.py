import numpy as np 
import os


def loadFile(fileName, directory=None, currFile=None):
	if directory != None:
		fileName= directory + '/' + fileName
	if currFile != None:
		fileName = absPath(currFile, fileName)

	return np.load(fileName)['_arr_0']


def absPath(currFile, fileName):
	return os.path.join(os.path.dirname, fileName)


def loadDirectory(directory, extName):
	filesList = filter(lambda f: f.endswith(extName) and os.path.isfile(os.path.join(directory, f)), os.listdir(directory))
	return map(lambda x: os.path.join(directory, x), filesList)



def removeExt(filename):
	return filename[:filename.find('.')]


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



def  meanImage(im_in):
	im_out = flattenImage(im_in)
	im_mean = np.mean(im_out, axis = 0)
	return im_mean


