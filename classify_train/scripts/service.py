#!/usr/bin/env python

import pickle
import numpy as np 
import rospy
import cv2
from sklearn import joblib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


clf = None

def loadModel(modelname):
	clf = pickle.dumps(modelname)
	return clf



def classifyCallback(req):
	print req.img;
	rosimg = req.img;
	cvbridge = CvBridge()
	try:
		cv_img = cvbridge.imgmsg_to_cv(rosimg, "bgr8")
	except CvBridgeError, e:
		print e

	sample = flattenImage(cv_img)
	preds = clf.predict(sample)
	


def flattenImage(image):
	n_rows, n_cols, _ = image.shape
	return image.reshape((n_rows*n_cols), 3)




if __name__ == '__main__':
	# model = loadModel('svm_classifier')
	clf = joblib.load('svm_classifier.pkl')

	rospy.init_node('color_classifier');
	rospy.Service('classify_color', Image_Transfer, classifyCallback)
	rospy.spin()
	

