# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 23:20:08 2018

@author: romai
"""

import math
import numpy as np
import matplotlib.pyplot as plt

class Contour:
    
    def __init__(self, points, shape):
        self.contourFromPoints(points, shape)
        self.npoints = len(self.points)
        self.computeNormals()
        
    def contourFromPoints(self, points, shape):
        array = np.zeros(shape)
        pts = []
        for c in range(len(points)):
            c1, c2 = points[c], points[(c + 1) % len(points)]
            for alpha in np.linspace(0, 1, 2*max(shape[0], shape[1])):
                x, y = int(c1[0] + alpha * (c2[0] - c1[0])), int(c1[1] + alpha * (c2[1] - c1[1]))
                if(len(pts) == 0):pts += [[x, y]]
                elif(pts[-1] != [x, y]):pts += [[x, y]]
                array[x, y] = 1
               
        interior = np.zeros(shape)
        
        for px in range(shape[0]):
            cur = array[px, 0]
            toAdd = np.zeros(shape)
            for py in range(shape[1]):
                if(array[px, py] == 1 and py != 0 and array[px, py - 1] == 0):
                    cur = (cur + 1) % 2
                toAdd[px, py] = cur
                
            if(cur != 1):
                interior += toAdd
                
        interior += array
        interior = np.minimum(interior, 1)
        
        self.array, self.interior, self.points = array, interior, np.array(pts)
    
    def computeNormals(self):
        normals = []
        for i in range(self.npoints):
            pm1, p, pp1 = self.points[(i - 1) % self.npoints], self.points[i], self.points[(i + 1) % self.npoints]
            if(pp1[0] != pm1[0] or pp1[1] != pm1[1]):ni = (pp1 - pm1).astype(float)
            else:ni = (p - pm1).astype(float)
            ni /= ((ni**2)**0.5).sum()
            ni = np.array([ni[1], -ni[0]])
            pfrompn = self.getPixelToNormal(p, ni)
            pfrommn = self.getPixelToNormal(p, - ni)
            
            print(self.npoints, i, ni, pm1, p, pp1, pfrompn, pfrommn, self.interior[pfrompn[0], pfrompn[1]], self.interior[pfrommn[0], pfrommn[1]])
            if(self.interior[pfrompn[0], pfrompn[1]] == 0):normals += [ni]
            elif(self.interior[pfrommn[0], pfrommn[1]] == 0):normals += [- ni]
            else:raise ValueError("PROBLEM")
            
        self.normals = np.array(normals)
        
    def getPixelToNormal(self, p, n):
        if(n[0] != 0):alpha = 180 * math.acos(n[1]) * np.sign(n[0]) / math.pi
        else:alpha = 180 * (np.sign(n[1]) < 0)
        while(alpha > 360):alpha -= 360
        while(alpha < 0): alpha += 360
        step = 360./16.
        if(alpha <= step or alpha > 15 * step):return p + np.array([0, 1]) #
        elif(alpha >= step and  alpha < 3 * step):return p + np.array([1, 1]) #
        elif(alpha >= 3 * step and  alpha < 5 * step):return p + np.array([1, 0]) #
        elif(alpha >= 5 * step and  alpha < 7 * step):return p + np.array([1, -1])
        elif(alpha >= 7 * step and  alpha < 9 * step):return p + np.array([0, -1])
        elif(alpha >= 9 * step and  alpha < 11 * step):return p + np.array([-1, -1]) #
        elif(alpha >= 11 * step and  alpha < 13 * step):return p + np.array([-1, 0]) #
        elif(alpha >= 13 * step and  alpha < 15 * step):return p + np.array([-1, 1])#
        else:raise ValueError("Alpha unvalid, alpha = " + str(alpha))
        
def testContour():
    
    shape = (50, 40)
    points = [[15, 20], [35, 5], [20, 33]]
    
    contour = Contour(points, shape)
    
    plt.subplot(121)
    plt.imshow(contour.array, cmap = "gray")
    plt.subplot(122)
    plt.imshow(contour.interior, cmap = "gray")
    plt.show()
    
    outside = np.zeros(shape)
    for i in range(contour.npoints):
        p = contour.getPixelToNormal(contour.points[i], contour.normals[i])
        outside[p[0], p[1]] = 1
    
    plt.imshow(2 * contour.array + outside, cmap = "gray")
    
    print(contour.points[:10])
    print(contour.normals[:10])
#testContour()