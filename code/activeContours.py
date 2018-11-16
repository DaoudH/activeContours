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
import cv2
from contour import Contour
from specHist import SpecHist
import matplotlib.pyplot as plt
import numpy as np
import yaml
PARAMS = yaml.load(open("params.yaml"))

class ActiveContours():
    
    def __init__(self, frames, path, spechist = False):
        self.path = path
        self.frames = self.preprocess(frames, spechist)
        if(len(self.frames[0].shape) == 2):
            self.shape = self.frames[0].shape
        elif(len(self.frames[0].shape) == 3):
            self.shape = self.frames[0].shape[:-1]
        else:
            raise ValueError("Frames shapes unvalid : " + str(self.frames[0].shape))
        
    def preprocess(self, frames, spechist = False):
        if(spechist):
            specification = SpecHist(frames[0])        
            return [cv2.blur(mycv2.resize(specification.specify(image), PARAMS["activeContours"]["maxwidth"]), (7, 7)) for image in frames]
        else:
            return [cv2.blur(mycv2.resize(image, PARAMS["activeContours"]["maxwidth"]), (7, 7)) for image in frames]
    
    def run(self):
        self.mask = Mask(self.frames[0])
        if(PARAMS["verbose"]):self.mask.render("contour")
        
        self.mask.computeDensity()
        if(PARAMS["debug"]):self.mask.render("density")
        
        self.currentcontour = self.mask.contour
        
        RESULT = []
        for i in range(0, len(self.frames)):
            RESULT += [self.computeSimpleFlow(i)]
        print(RESULT[0].shape, "SHAPE   ", vid.getFPS(self.path), "FPS")
        print([r.shape for r in RESULT])
        
        shape = (int(RESULT[0].shape[0] * 2), int(RESULT[0].shape[1]))
        print(shape)
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'),
                              vid.getFPS(self.path), (512, 512))
        
        print(vid.getFPS(self.path), shape)
        for f, res in zip(self.frames, RESULT):
            res3 = np.array([res, res, res]).transpose(1, 2, 0) * 255
            newf = cv2.resize(np.uint8(np.concatenate([f, res3], axis = 0)), (512, 512))
            #plt.imshow(newf[:, :, ::-1])
            #plt.show()
            
            out.write(newf)
                
        out.release()
        
    def computeSimpleFlow(self, i):
        print("computeSimpleFlow", i, "     ", int(100 * i / len(self.frames)), "%")
        frame = self.frames[i]
        #plt.imshow(frame[:, :, ::-1])
        #plt.show()
        """
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
        
        def getPiDC(pi, ni, nax):
            
            DC = np.zeros(2 * nax + 1)
            Pi = np.zeros((2 * nax + 1, 2))
            
            DC[nax] = np.sign(self.mask.getDensity(frame[pi[0], pi[1]]) - self.mask.lambd)
            Pi[nax] += pi
            
            for j in range(1, nax + 1):
                temppi = self.currentcontour.getPixelToNormal(Pi[nax + j - 1], ni).copy().astype(int)
                Pi[nax + j] = temppi.copy()
                tempdc = self.mask.getDensity(frame[temppi[0] % self.shape[0], temppi[1] % self.shape[1]]) - self.mask.lambd
                DC[nax + j] += np.sign(tempdc)
                
                temppi = self.currentcontour.getPixelToNormal(Pi[nax - j + 1], - ni).copy().astype(int)
                Pi[nax - j] = temppi.copy()
                tempdc = self.mask.getDensity(frame[temppi[0] % self.shape[0], temppi[1] % self.shape[1]]) - self.mask.lambd
                DC[nax - j] += np.sign(tempdc)
            
            return Pi, DC
        
        
        points = self.currentcontour.get("points").copy()
        normals = self.currentcontour.get("normals").copy()
        print(len(points))
        findeps = []
        #findepm1 = 0
        changepoints = []
        #for i in range(len(newpoints)):
        nbloc = 10
        nax = 25
        nnewpoints = max(25, int(0.1 * len(points)))
        for i in np.linspace(0, len(points), nnewpoints + 1).astype(int)[:-1]:
            findepii = []
            for ii in range(-nbloc, nbloc + 1):
                pi = points[(i + ii) % len(points)].copy()
                ni = normals[(i + ii) % len(normals)].copy()
                
                Pi, DC = getPiDC(pi, ni, nax)
                if(ii == 0):
                    Pi0 = Pi.copy()
                
                dep = []
                for j in range(nax):
                    if(DC[j] < 0 and DC[j + 1] > 0):
                        dep += [j + 1]
                        
                for j in range(1, nax + 1):
                    if(DC[nax + j] < 0 and DC[nax + j - 1] > 0):
                        dep += [nax + j - 1]
            
                if(np.sum(DC) == len(DC)):
                    dep += [len(DC) - 1]
                if(np.sum(DC) == -len(DC)):
                    dep += [0]
                if(len(dep) == 0):
                    dep += [nax]
                
                dep = np.array(dep)
                if(np.sum(dep < 25) > np.sum(dep > 25)):
                    dep = dep[dep < 25]
                elif(np.sum(dep < 25) < np.sum(dep > 25)):
                    dep = dep[dep > 25]
                else:
                    dep = [25]
                    
                findepii += [np.median(dep)]
                
            findep = int(np.median(findepii))
            
            """
            if(i != 0):
                if(findep > findepm1 + 5):
                    findep = findepm1 + 5
                elif(findep < findepm1 - 5):
                    findep = findepm1 - 5
            findepm1 = findep
            """
            
            findeps += [findep]
            #newpoints[i] = Pi[findep].copy()
            changepoints += [Pi0[findep].copy()]
            """
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
            """
        #plt.plot(np.arange(len(findeps)), np.array(findeps) - 25)
        #plt.show()
        self.currentcontour = Contour(changepoints, self.shape)
        
        if(PARAMS["verbose"]):self.currentcontour.render()
        return self.currentcontour.interior.copy()