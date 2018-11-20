## @package video
# Video methods
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 22:42:03 2018

@author: romai
"""
import cv2

## Stack all frames from a video.
# @param path The path to the video.
# @return A list of opencv images.
def getFrames(path):
    vidcap = cv2.VideoCapture(path) #Initializes the opencv object
    success,image = vidcap.read()   #Reads the video
    frames = [image]    #Gets the frames
    while success:  #Read the frames frame by frame
        success,image = vidcap.read()
        frames += [image]   
    return frames[:-1] #Returns the frames

## Gets the number of frame per seconds of a video.
# @param path The path to the video.
# @return A float corresponding to the number of FPS in a video.
def getFPS(path):
    vidcap = cv2.VideoCapture(path) #Initializes the opencv object
    return vidcap.get(cv2.CAP_PROP_FPS) 