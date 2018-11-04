# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 23:20:08 2018

@author: romai
"""

import numpy as np

class Contour:
    
    def __init__(self, points, shape):
        self.array = self.contourFromPoints(points, shape)
        
    def contourFromPoints(self, points, shape):
        array = np.zeros(shape)
        for c in range(len(points)):
            c1, c2 = points[c], points[(c + 1) % len(points)]
            for alpha in np.linspace(0, 1, 2*max(shape[0], shape[1])):
                array[int(c1[0] + alpha * (c2[0] - c1[0])), int(c1[1] + alpha * (c2[1] - c1[1]))] = 1
               
        return array.transpose()