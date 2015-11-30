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
import matplotlib.pyplot as plt
import time

clf = None



ap = argparse.ArgumentParser();

ap.add_argument('-i', '--image')
args = vars(ap.parse_args());
image = cv2.imread(args["image"])
# print image.shape 
# exit()

classes = {0:"green_dark", 1:"red", 2:"blue", 3:"orange", 4:"green_light", 5:"violet", 6:"yellow"}

# blobParams = cv2.SimpleBlobDetector_Params()
# detector 


def setBlobDetector():
	blobParams.minThreshold = 20
	blobParams.maxThreshold = 250
	blobParams.minDistBetweenBlobs = 40
	# Filter by Area.
	blobParams.filterByArea = True
	blobParams.minArea = 350

	# Filter by Circularity
	blobParams.filterByCircularity = True
	blobParams.minCircularity = 0.15
	blobParams.filterByConvexity = True
	blobParams.minConvexity = 0.72
    
	# Filter by Inertia
	blobParams.filterByInertia = True
	blobParams.minInertiaRatio = 0.08
	detector = cv2.SimpleBlobDetector(blobParams)




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



def shapeCallback(req):
	rosimg = req.im;
	cvbridge = CvBridge()
	try:
		bw_img = cvbridge.imgmsg_to_cv2(rosimg, desired_encoding="passthrough")
	except CvBridgeError, e:
		print e

	print bw_img.shape
	# fin_bw_img = 
	ts = time.time()
	name = 'green_hol_cyl_bw'  + '.png'
	# fig1 = plt.figure()
	# plt.imshow(fig1)
	np.savetxt('green_hol_cyl_bw.txt', bw_img)


	# plt.savefig(name, bw_img)

	



def predictClass(predictions):
	votecount = Counter(predictions)
	cat =  votecount.most_common(1)
	counts = np.bincount(predictions, minlength = 7)
	counts = counts.tolist()

	if len(counts) < 7:
		diff = 7 - len(counts)
		bufferlist = []
		val = 0
		for i in xrange(diff):
			bufferlist.append(val)
		counts = counts.extend(bufferlist)
	color = decisionRules(counts)
	# print counts
	# return np.argmax(counts)
	return color



def flattenImage(image, channels=2):
	n_rows, n_cols, n_channels = image.shape
	newimg = image[:,:,:2]
	# print newimg[0:10]
	return newimg.reshape((n_rows*n_cols), channels)

def bwImage(image, channels=2):
	n_rows, n_cols, n_channels = image.shape
	newimg[:,:,0] = (image[:,:,0] + image[:,:,1] + image[:,:,2])/3
	plt.imshow(newimg) 
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


#  this is a bit hacky.
def decisionRules(counts):

	# @TODO:investigate blue  
	# normalize for blue color, i dont know why the classififer is so biased towards blue.
	if counts[2] > 300:
		counts[2] = counts[2] - 300
	print counts


	maxVote = max(counts)



	if maxVote < 200:
		# return "something"
		return 99


	if counts[3] >= 70:
		# return classes[3]
		return 3


	if counts[4] >= 130 and counts[0] < 800:
		# return classes[4]
		return 4


	if counts[1] >= 500:
		# return classes[1]
		return 1

	# treat light green and dark green as the same, as the votes possibly get split among the colors.
	greenVotes = counts[0] + counts[4]

	# orange and yellow are also similar
	oryeVotes = counts[3] + counts[6]


	if greenVotes > maxVote:
		if counts[0] >= counts[4]:
			# return classes[0]
			return 0
		else:
			# return classes[4]
			return 4

	if oryeVotes > maxVote:
		if counts[3] >= counts[6]:
			# return classes[3]
			return 3
		else:
			# return classes[6]
			return 6

	color = np.argmax(counts)
	# return classes[color] 
	return color






if __name__ == '__main__':
	# model = loadModel('svm_classifier')
	# clf = joblib.load('svm_classifier.pkl')
	clf = pickle.load(open("svm_classifier_linear.p","rb"))
	# clf = pickle.load(open("svm_classifier_rbf1.p","rb"))

	# image = 
	# testClassifier(image)

	rospy.init_node('color_classifier');
	rospy.Service('/classify_objects/color', Image_Transfer, classifyCallback)
	# rospy.Service('/classify_objects/shape', Image_Transfer, shapeCallback)
	rospy.spin()
	

