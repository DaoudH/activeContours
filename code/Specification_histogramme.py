# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:15:21 2018

@author: HP
"""

def specif(video_path):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    video=cv2.VideoCapture(video_path)
    _,firstimage=video.read()
    height,width,layers=firstimage.shape
    firstimage=cv2.cvtColor(firstimage, cv2.COLOR_BGR2GRAY)
    t=0
    while True:

        _,frame=video.read()
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        v = firstimage
        u = frame
        [nrowu,ncolu]=u.shape
        [nrowv,ncolv]=v.shape
        u_sort,index_u=np.sort(u,axis=None),np.argsort(u,axis=None)
        [v_sort,index_v]=np.sort(v,axis=None),np.argsort(v,axis=None)
        uspecifv= np.zeros(nrowu*ncolu)
        uspecifv[index_u] = v_sort
        uspecifv = uspecifv.reshape(nrowu,ncolu)
        #cv2.imshow("Frame",frame)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        axes[0].set_title('Image t='+str(t))
        axes[0].imshow(u,'gray')
        axes[1].set_title('First Image')
        axes[1].imshow(v,'gray')
        axes[2].set_title('Image t='+str(t)+ '+ specification')
        axes[2].imshow(uspecifv,'gray')
        t+=1
    cv2.destroyAllWindows()
    video.release()



