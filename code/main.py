# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:19:47 2018

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

def main():
    image = mycv2.loadImage(PARAMS["data"]["images"]["lena"])
    mycv2.show(image)
    
    mask = Mask(image)
    
    plt.imshow(mask.mask, cmap = "gray")
    plt.show()
    
main()