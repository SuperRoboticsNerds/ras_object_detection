#!/usr/bin/env python

import numpy as np 
import os, sys

import rospy
import cv2

from sklearn import svm
from sklearn import cross_validation
from sklearn.multiclass  import  OneVsRestClassifier
import pickle
from sklearn.externals import joblib
import matplotlib.pyplot as plt





def errorRate(estimates, actual):
	# estimates = np.transpose(estimates).tolist()
	# print estimates
	# actual = np.transpose(actual).tolist()
	# actual = actual[0]
	# estimates = np.array(estimates)
	# actual = np.array(actual)
	# print estimates != actual
	return float(sum(estimates != actual)) / float(actual.shape[0]) 



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
			# print error_rate

	else:
		for train_ind, test_ind in s:
			cls = svm.SVC(C = 100000, gamma =0.1, kernel='rbf', verbose= False)
			X_train = data_x[train_ind]
			Y_train = data_y[train_ind]
			X_test = data_x[test_ind]
			Y_test = data_y[test_ind]
			cls.fit(X_train, Y_train)

			# dd = np.column_stack((gg[0].flat, gg[1].flat))

			preds = cls.predict(X_test)
			error_rate = errorRate(preds, Y_test)
			print error_rate

	return cls


filesList = ['material_data/hollow_mat.txt', 'material_data/solid_data.txt']

d1 = np.loadtxt('material_data/hollow_mat.csv', delimiter=',')
d2 = np.loadtxt('material_data/solid_mat.csv', delimiter=',')
# d2 = np.loadtxt('material_data/hollow_mat.txt', delimiter='/t', dtype = 'float32')

# data = map(lambda x: np.loadtxt(x, delimiter='/t', dtype='float32'), filesList)

n_hollow = d1.shape[0]
n_solid = d2.shape[0]

n_feats = min(n_hollow, n_solid)
d1 = d1[:n_feats]
d2 = d2[:n_feats]

data_x = np.concatenate((d1, d2), axis = 0)

data_y0 = [0] * n_feats
data_y1 = [1] * n_feats

data_y = np.concatenate((data_y0, data_y1), axis=0)

s = cross_validation.ShuffleSplit(len(data_y), 3)

clf = trainClassifier(s)

pickle.dump(clf, open("svm_clf_material.p", "wb"))
