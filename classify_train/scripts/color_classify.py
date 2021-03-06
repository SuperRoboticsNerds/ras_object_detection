#!/usr/bin/env python

import numpy as np 
import os , sys

# import  classify_train

import rospy
import cv2
from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass  import  OneVsRestClassifier
import pickle
from sklearn.externals import joblib
import matplotlib.pyplot as plt


# filesList = ['color_data/green_cube.npz', 'color_data/red_cube.npz',  \
# 'color_data/blue_prism.npz', 'color_data/orange_star.npz',  \
#  'color_data/green_light_cube.npz', 'color_data/violet_cross.npz', 'color_data/yellow_cube.npz'] 


# filesList = ['color_data1/green_cube1.npz', 'color_data1/red_cube1.npz',  'color_data1/blue_tri1.npz',  \
#  'color_data1/orange_star1.npz',  'color_data1/green_cyl1.npz', 'color_data1/purple_cross1.npz',
#   'color_data1/yellow_cube1.npz'] 


filesList = ['color_data2/green_cube2.npz', 'color_data2/red_cube2.npz',  'color_data2/blue_tri2.npz',  \
 'color_data2/orange_star2.npz',  'color_data2/green_cyl2.npz', 'color_data2/purple_cross2.npz',
  'color_data2/yellow_cube2.npz'] 


clf = None

def loadDirectory(directoryname):
	fileList = map(lambda x: os.path.join(directoryname, x) , os.listdir(directoryname))
	return dataList


def absPath(filename):
	return os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
	# return os.path.join(os.path.dirname, filename)


def loadData(filename):
	feat_data = np.load(filename)['arr_0']
	return feat_data


def dropData(in_data, num_rows, num_cols =2, drop_percent= 0.90):
	out_data = in_data[:num_rows]
	tot_rows = abs(num_rows*(1 - drop_percent))
	out_data = out_data[:tot_rows,:num_cols]
	return out_data



def subSampleData(dataList):
	sizes = [d.shape[0] for d in dataList]
	min_size = min(sizes)
	print min_size
	# norm_data_list = map(lambda x : dropData(x, min_size, 2, 0.50), dataList)	

	norm_data_list = map(lambda x : dropData(x, min_size, 2, 0.50), dataList)	
	return norm_data_list


def errorRate(estimates, actual):
	return float(sum(estimates != actual)) / float(len(actual)) 



def trainClassifier(s, linear = 0):
	if linear == 1:
		for train_ind, test_ind in s:
			cls = svm.LinearSVC()
			X_train = data_x[train_ind]
			Y_train = data_y[train_ind]
			X_test = data_x[test_ind]
			Y_test = data_y[test_ind]
			cls.fit(X_train, Y_train)

			# dd = np.column_stack((gg[0].flat, gg[1].flat))

			preds = cls.predict(X_test)
			error_rate = errorRate(preds, Y_test)
			print error_rate

	else:
		for train_ind, test_ind in s:
			cls = svm.SVC(C = 10, gamma =0.1, kernel='rbf', verbose= False)
			X_train = data_x[train_ind]
			Y_train = data_y[train_ind]
			X_test = data_x[test_ind]
			Y_test = data_y[test_ind]
			cls.fit(X_train, Y_train)

			# dd = np.column_stack((gg[0].flat, gg[1].flat))

			d1 = np.meshgrid(xrange(0, 181), xrange(0,256))
			d2 = np.column_stack((d1[0].flat, d1[1].flat))

			all_preds = cls.predict(d2)
			pred_reshaped = all_preds.reshape(256, 181)
			pickle.dump(pred_reshaped, open("look_up_real2.p", "wb"))
				
			preds = cls.predict(X_test)
			error_rate = errorRate(preds, Y_test)
			print error_rate

	return cls





def saveModel(clfname, out_file):
	pass


def callback():
	pass



if __name__ == '__main__':

	filesAbsList = [absPath(f) for f in filesList]
	dataList = [loadData(f) for f in filesAbsList]
	C = len(dataList)
	# print C
	# exit()
	norm_data  = subSampleData(dataList)
	# print len(norm_data)
	data_x = np.concatenate(norm_data)
	print data_x.shape
	print data_x[1:10]
	print data_x[60000:60010]
	# exit()
	sample_size = norm_data[0].shape[0]

	# labels = [np.repeat(x, sample_size) for x in xrange(0, C)]	
	# labels = map(lambda x: np.repeat([x], sample_size), range(0,C))
	labels = map(lambda (f, l): np.repeat(l, f), zip(map(len, norm_data), xrange(0, C)))

	data_y = np.concatenate(labels)
	print data_x.shape, data_y.shape

	s = cross_validation.ShuffleSplit(len(data_y), 1)

	clf = trainClassifier(s)

	
	# pickle.dump(cls, open("svm_classifier", "wb"))
	# pickle.dumps(cls)
	pickle.dump(clf, open("svm_classifier_rbf_real2.p", "wb"))
	# clf = pickle.load(open("svm_classifier_linear.p","rb"))

		# joblib.dump(cls, 'svm_classifier.pkl')


