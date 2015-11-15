
import numpy as np 
import os , sys


import rospy
import cv2
from sklearn import svm
from sklearn import cross_validation
from time import time 



filesList = ['training_data1/blue_cube.npz', 'training_data1/green_cube.npz', 'training_data1/orange_cross.npz', 'training_data1/red_cube.npz']


def loadData(filename):
	feat_data = np.load(filename)['arr_0']
	return feat_data

def dropData(in_data, num_rows, drop_percent):
	out_data = in_data[:num_rows]
	tot_rows = abs(num_rows*(1 - drop_percent))
	# // so lame
	out_data = out_data[:tot_rows]
	return out_data



def subSampleData(dataList):
	sizes = [d.shape[0] for d in dataList]
	min_size = min(sizes)
	norm_data_list = map(lambda x : dropData(x, min_size, 0.25), dataList)	
	return norm_data_list


def errorRate(estimates, actual):
	return sum(estimates != actual)/ float(len(actual)) 


if __name__ == '__main__':

	dataList = [loadData(f) for f in filesList]
	C = len(dataList)
	print dataList[0]
	# exit()
	norm_data  = subSampleData(dataList)
	data_x = np.concatenate(norm_data)
	sample_size = norm_data[0].shape[0]
	# labels = [np.repeat(x, sample_size) for x in xrange(0, C)]	
	# labels = map(lambda x: np.repeat([x], sample_size), range(0,C))
	labels = map(lambda (f, l): np.repeat(l, f), zip(map(len, norm_data), xrange(0, C)))

	data_y = np.concatenate(labels)
	print data_x.shape, data_y.shape

	s = cross_validation.ShuffleSplit(len(data_y), 3)

	for train_ind, test_ind in s:
		cls = svm.SVC(C=10, kernel='rbf', degree = 1.0, verbose=False )
		X_train = data_x[train_ind]
		Y_train = data_y[train_ind]
		X_test = data_x[test_ind]
		Y_test = data_y[test_ind]
		cls.fit(X_train, Y_train)

		gg = np.meshgrid(xrange(0, 180, 1), xrange(0, 256, 2), xrange(0, 256, 2))
		dd = np.column_stack(gg[0].flat, gg[1].flat, gg[2].flat)

		preds = cls.predict(X_test)
		error_rate = errorRate(preds, Y_test)

		preds_sp = cls.predict()




		print error_rate




	







