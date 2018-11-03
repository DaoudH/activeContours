# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 22:50:34 2018

@author: Hasnaa Daoud
"""

import cv2
import numpy as np
video=cv2.VideoCapture("../test.mp4") #Path à modifier
_,firstimage=video.read() #recupérer la première image de la vidéo
x=195 #coordonnée x du point gauche haut  de l'objet à suivre
y=120#coordonnée y du point gauche haut de l'objet à suivre 
width=100 #largeur de l'objet
height=100 #hauteur de l'objet
objet=firstimage[y:y+height,x:x+width]
hsv_objet=cv2.cvtColor(objet,cv2.COLOR_BGR2HSV)
objet_hist=cv2.calcHist([hsv_objet],[0],None,[180],[0,180])
objet_hist=cv2.normalize(objet_hist,objet_hist,0,255,cv2.NORM_MINMAX)
#print(objet_hist)
term_criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1 )
while True:
    _,frame=video.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.calcBackProject([hsv],[0],objet_hist,[0,180],1)
    _,track_window=cv2.meanShift(mask,(x,y,width,height),term_criteria)
    x,y,w,h=track_window
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("objet",objet)
    cv2.imshow("mask",mask)
    cv2.imshow("Frame",frame)
    cv2.imshow("FirstImage",firstimage)
    key=cv2.waitKey(60)
    if key==27:
        break
video.release()
cv2.destroyAllWindows()
