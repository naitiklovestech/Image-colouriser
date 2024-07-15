#import all the required libraries
import numpy as np
import argparse
import cv2
import os

# paths to load the model

#Load the model
print("Loading the model")
net = cv2.dnn.readNetFromCaffe(PROTOTEXT, MODEL)
pts = np.load(POINTS)

#Load channel for ab center quantisation used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1,1)
net.getlayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313], 2.606, dtype="float32")]

#Load the input image
image = cv2.imread()