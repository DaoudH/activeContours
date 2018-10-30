## @package opencv_utils
# Encapsulates opencv functions to use them quicker in our project.
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:19:26 2018

@author: romai
"""

import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
verbose = yaml.load(open("params.yaml"))["verbose"]
PARAMS = yaml.load(open("params.yaml"))

## Show an image.
# @param image The image.
# @param maxwidth The maximum width of the image.
# This function of course conserves proportions.
def show(image, maxwidth = np.nan):
    if(PARAMS["render"]["quick"]):
        plt.imshow(image)
        plt.show()
    else:
        if(np.isnan(maxwidth)):
            maxwidth = PARAMS["render"]["maxwidth"]
        cv2.imshow('image',cv2.resize(image, (maxwidth, int(maxwidth * image.shape[0] / image.shape[1]))))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

## Load an image.
# @param path The path to the classifier.
# @return The image.
def loadImage(path):
    if(verbose):
        print("Loading image from " + path + " ... ", end = "")
    image = cv2.imread(path) #Load image
    if(verbose):
        print("Done")
    return image

## Convert an RGB image in a grayscale image.
# @param image The image.
# @return The grayscaled image.
def cvtGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convert to gray