# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 21:59:40 2018

@author: romai
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def createVid():
    shape = (512, 256)
    
    out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'XVID'), 30., shape)
    
    Cx = np.concatenate([np.linspace(256, 100, 100),
                         np.linspace(100, 150, 100),
                         np.linspace(150, 400, 100)])
    Cy = np.concatenate([np.linspace(128, 75, 100),
                         np.linspace(75, 200, 100),
                         np.linspace(200, 75, 100)])
    C = np.array([Cx, Cy]).transpose()
    col1 = np.random.random(3)
    col1 /= np.sum(col1)
    col2 = np.random.random(3)
    col2 /= np.sum(col2)
    
    r = 50
    
    def createArr(c, r):
        arr = np.array(np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))).transpose(1, 2, 0) + 5 * (2 * np.random.random((256, 512, 2)) - 1)
        arr = (np.sum((arr - c)**2, axis = -1) < r**2).astype(float)
        arr = np.array([arr, arr, arr]).transpose(1, 2, 0)
        arr = col1 * arr + col2 * (1 - arr)
        arr += .05 * (2 * np.random.random(arr.shape) - 1)
        arr = np.maximum(np.minimum(arr, 1.), 0.)

        return np.uint8(255 * arr)
    
    plt.imshow(createArr(C[0], r))
    plt.show()
    
    for c in C:
        arr = createArr(c, r)
        out.write(arr)
         
    out.release()
        
createVid()