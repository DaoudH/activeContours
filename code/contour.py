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
        self.shape= shape
        self.npoints = len(self.points)
        self.computeNormals()
        
    def checkPoints(self, points):
        newpoints = []
        for c in range(len(points)):
            c1, c2 = points[c], points[(c + 1) % len(points)]
            if(c1[0] != c2[0] or c1[1] != c2[1]):
                newpoints += [c1]
                
        return newpoints
        
    def contourFromPoints(self, points, shape):
        points = self.checkPoints(points)
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
            
            #print(self.npoints, i, ni, pm1, p, pp1, pfrompn, pfrommn, self.interior[pfrompn[0], pfrompn[1]], self.interior[pfrommn[0], pfrommn[1]])
            if(self.interior[pfrompn[0] % self.shape[0], pfrompn[1] % self.shape[1]] == 0):normals += [ni]
            elif(self.interior[pfrommn[0] % self.shape[0], pfrommn[1] % self.shape[1]] == 0):normals += [- ni]
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
        
    def get(self, param):
        if(param == "points"):return self.points
        elif(param == "interior"):return self.interior
        elif(param == "border_in"):
            border_in = []
            for i in range(self.npoints):
                p = self.getPixelToNormal(self.points[i], - self.normals[i])
                if(not self.array[p[0], p[1]] and self.interior[p[0], p[1]]):
                    border_in += [p]
            return Contour(border_in, self.shape)
        elif(param == "border_out"):
            border_out = []
            for i in range(self.npoints):
                p = self.getPixelToNormal(self.points[i], self.normals[i])
                if(not self.array[p[0], p[1]] and not self.interior[p[0], p[1]]):
                    border_out += [p]
            return Contour(border_out, self.shape)
        else:raise ValueError("param unvalid for contour getter, param = " + str(param))
        
    def render(self):
        plt.subplot(121)
        plt.imshow(self.array, cmap = "gray")
        plt.subplot(122)
        plt.imshow(self.interior, cmap = "gray")
        plt.show()
        
def testContour():
    
    shape = (50, 50)
    points = [[15, 20], [35, 5], [20, 33]]
    
    contour = Contour(points, shape)
    contour.render()
    
    contour.get("border_out").render()
    contour.get("border_in").render()

#testContour()