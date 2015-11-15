#!/usr/bin/env python

import pickle
import numpy as np 
import rospy
import cv2
import argparse
from sklearn.externals import joblib
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ras_object_lib.srv import Image_Transfer
from ras_object_lib.srv import Image_TransferResponse
from collections import Counter


clf = None



ap = argparse.ArgumentParser();

ap.add_argument('-i', '--image')
args = vars(ap.parse_args());
image = cv2.imread(args["image"])
# print image.shape 
# exit()

classes = {0:"green", 1:"red", 2:"blue", 3:"orange", 4:"green_light", 5:"violet", 6:"yellow"}


def loadModel(modelname):
	clf = pickle.dumps(modelname)
	return clf


def classifyCallback(req):
	# print req.im;
	rosimg = req.im;
	cvbridge = CvBridge()
	try:
		cv_img = cvbridge.imgmsg_to_cv2(rosimg, desired_encoding="passthrough")
	except CvBridgeError, e:
		print e


	hsv_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
	# sample = flattenImage(cv_img)
	sample = flattenImage(hsv_image, 2)

	preds = clf.predict(sample)
	label = predictClass(preds)
	return Image_TransferResponse(str(label))
	# return Image_TransferResponse(str(classes[label]))



def predictClass(predictions):
	votecount = Counter(predictions)
	cat =  votecount.most_common(1)
	counts = np.bincount(predictions)
	print counts
	return np.argmax(counts)
	# return cat



def flattenImage(image, channels=2):
	n_rows, n_cols, n_channels = image.shape
	newimg = image[:,:,:2]
	# print newimg[0:10]
	return newimg.reshape((n_rows*n_cols), channels)



def testClassifier(img):
	hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# img1 = flattenImage(img)
	img1 = flattenImage(hsv_image)
	preds = clf.predict(img1)	
	print preds
	label = predictClass(preds)
	print str(classes[label])




if __name__ == '__main__':
	# model = loadModel('svm_classifier')
	# clf = joblib.load('svm_classifier.pkl')
	clf = pickle.load(open("svm_classifier_linear.p","rb"))

	# image = 
	# testClassifier(image)

	rospy.init_node('color_classifier');
	rospy.Service('/classify_objects/color', Image_Transfer, classifyCallback)
	rospy.spin()
	

