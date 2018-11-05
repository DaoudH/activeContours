# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:11:21 2018

@author: romai
"""

import sys
sys.path.append('utils')

from mask import Mask
import mycv2 as mycv2
import video as vid
import cv2 as cv2
from contour import Contour
import matplotlib.pyplot as plt
import numpy as np
import yaml
PARAMS = yaml.load(open("params.yaml"))

class ActiveContours():
    
    def __init__(self, frames):
        self.frames = self.preprocess(frames)
        self.shape = self.frames[0].shape[:-1]
        
    def preprocess(self, frames):
        return [cv2.blur(mycv2.resize(image, PARAMS["activeContours"]["maxwidth"]), (5, 5)) for image in frames]
    
    def run(self):
        self.mask = Mask(self.frames[0])
        if(PARAMS["verbose"]):self.mask.render("contour")
        
        self.mask.computeDensity()
        if(PARAMS["debug"]):self.mask.render("density")
        
        self.currentcontour = self.mask.contour
        
        RESULT = []
        for i in range(0, len(self.frames)):
            RESULT += [self.computeSimpleFlow(i)]
        
    def computeSimpleFlow(self, i):
        frame = self.frames[i]
                
        FD = np.zeros(self.shape)
        for px in range(self.shape[0]):
            for py in range(self.shape[1]):
                FD[px, py] = self.mask.getDensity(frame[px, py])
        plt.figure(figsize = (10, 5))
        plt.subplot(121)
        plt.imshow(FD, cmap = "gray")
        plt.subplot(122)
        plt.imshow(FD > 0, cmap = "gray")
        plt.show()
         
        """
        while(nchanges > 0 and nite < 10):
            nadded, nremoved = 0, 0
            newpoints = self.currentcontour.get("points").copy()
            normals = self.currentcontour.get("normals").copy()
            
            for i in range(len(newpoints)):
                dc = self.mask.getDensity(frame[newpoints[i][0], newpoints[i][1]]) - self.mask.lambd
                if(dc < 0):
                    newpoints[i] = self.currentcontour.getPixelToNormal(newpoints[i], - normals[i]).copy()
                    nremoved += 1
                else:    
                    p_n = self.currentcontour.getPixelToNormal(newpoints[i], normals[i]).copy()
                    dc_p_n = self.mask.getDensity(frame[p_n[0], p_n[1]]) - self.mask.lambd
                    if(dc_p_n > 0):
                        newpoints[i] = p_n.copy()
                        nadded += 1
            self.currentcontour = Contour(newpoints, self.shape)
            nite += 1
            nchanges = nadded + nremoved
            print(nite, nadded, nremoved, len(self.currentcontour.get("points")))
            #if(PARAMS["verbose"]):self.currentcontour.render()
            #if(PARAMS["verbose"]):self.currentcontour.render()
        """
        
        newpoints = self.currentcontour.get("points").copy()
        normals = self.currentcontour.get("normals").copy()
        
        for i in range(len(newpoints)):
            pi = newpoints[i].copy()
            ni = normals[i].copy()
            dc = self.mask.getDensity(frame[pi[0], pi[1]]) - self.mask.lambd
            nite = 0
            if(dc < 0):
                while(dc < 0 and nite < 50):
                    pi = self.currentcontour.getPixelToNormal(pi, - ni).copy() 
                    dc = self.mask.getDensity(frame[pi[0] % self.shape[0], pi[1] % self.shape[1]]) - self.mask.lambd
                    nite += 1
            else:
                while(dc > 0 and nite < 50):
                    pi = self.currentcontour.getPixelToNormal(pi, ni).copy() 
                    dc = self.mask.getDensity(frame[pi[0], pi[1]]) - self.mask.lambd
                    nite += 1
            if(nite < 50):newpoints[i] = pi.copy()
        print(np.sum(np.abs(newpoints - self.currentcontour.get("points"))))
        self.currentcontour = Contour(newpoints, self.shape)
        if(PARAMS["verbose"]):self.currentcontour.render()
        return self.currentcontour