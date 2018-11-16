# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:16:14 2018

@author: romai
"""

import sys
sys.path.append('utils')

from mask import Mask
from activeContours import ActiveContours
import mycv2 as mycv2
import video as vid
import matplotlib.pyplot as plt
import yaml
PARAMS = yaml.load(open("params.yaml"))

def demoMaskCreation():
    image = mycv2.loadImage(PARAMS["data"]["images"]["lena"])
    image = mycv2.cvtGray(image)
    mycv2.show(image)
    
    mask = Mask(image)
    
    mycv2.show(mask.mask)
    mycv2.show(mask.hull)
    print("Area", mask.area())
    
#demoMaskCreation()

def demoVideo():
    path = PARAMS["data"]["video"]["movingsquare"]
    frames = vid.getFrames(path)
    mycv2.show(frames[0])
    
    algo = ActiveContours(frames, path, spechist = False)
    algo.run()
    
demoVideo()