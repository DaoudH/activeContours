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
import matplotlib.pyplot as plt
import yaml
PARAMS = yaml.load(open("params.yaml"))

class ActiveContours():
    
    def __init__(self, frames):
        self.frames = self.preprocess(frames)
        
    def preprocess(self, frames):
        return [mycv2.resize(image, PARAMS["activeContours"]["maxwidth"]) for image in frames]
    
    def run(self):
        self.mask = Mask(self.frames[0])
        
        