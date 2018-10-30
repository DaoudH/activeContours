# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 18:55:13 2018

@author: romai
"""

import imp
        
def checkModule(name):
    try:
        imp.find_module(name)
    except ImportError:
        print("You don't have the module " + name + ". You should consider installing it in order for this project to work.")
        print("If you have trouble installing it, go check the wiki :", "https://github.com/mathsForBusinessUbble2018/trueFaceDetection/wiki/HELP-ME-!")
    
def checkModules():
    for module in ["numpy", "matplotlib", "cv2", "yaml", "easygui", "tkinter"]:
        checkModule(module)

checkModules()

