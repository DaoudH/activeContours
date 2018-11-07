# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:15:21 2018

@author: HP
"""
import sys
sys.path.append('utils')
import cv2
import mycv2 as mycv2
import numpy as np
import video as vid
import matplotlib.pyplot as plt
import yaml
PARAMS = yaml.load(open("params.yaml"))

class SpecHist:
    
    def __init__(self, frame):
        self.frame = frame
        self.gray = mycv2.cvtGray(frame)
        self.nrow, self.ncol = self.gray.shape
        self.sort, self.index = np.sort(self.gray, axis=None), np.argsort(self.gray, axis=None)
        
    def specify(self, newFrame):
        newGray = mycv2.cvtGray(newFrame)
        newNrow, newNcol = newGray.shape
        
        newSort, newIndex = np.sort(newGray, axis=None), np.argsort(newGray, axis=None)
        
        specNew = np.zeros(newNrow * newNcol)
        specNew[self.index] = newSort
        specNew = specNew.reshape(newNrow, newNcol)
        
        return specNew
        
def runTests():
    frames = vid.getFrames(PARAMS["data"]["video"]["movingcable"])

    spec = SpecHist(frames[0])
    newFrame1 = spec.specify(frames[1])

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
    axes[0].set_title('Image')
    axes[0].imshow(frames[0],'gray')
    axes[1].set_title('Image')
    axes[1].imshow(frames[1],'gray')
    axes[2].set_title('Specification')
    axes[2].imshow(newFrame1,'gray')

#runTests()