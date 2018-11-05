# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:16:07 2018

@author: romai
"""

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import mycv2 as mycv2
from contour import Contour
import cv2 as cv2
import matplotlib.pyplot as plt
import yaml
PARAMS =  yaml.load(open("params.yaml"))
verbose = yaml.load(open("params.yaml"))["verbose"]
debug = yaml.load(open("params.yaml"))["debug"]

class Mask:
    
    def __init__(self, image, contour = np.nan):
        if(len(image.shape) == 2):
            self.n = 1
        else:
            self.n = image.shape[-1]
        self.image = image
        if(np.isnan(contour)):
            self.contour = Windows(self.image, self.n).contour
        else:
            self.contour = contour
    
    def area(self):
        return self.contour.interior.sum() / (self.contour.shape[0] * self.contour.shape[0])
    
    #WARNING, this function is very costly, consider using self.computeDensity()
    def N(self, z):
        N = 0
        for px in range(self.image.shape[0]):
            for py in range(self.image.shape[1]):
                if(self.contour.interior[px, py]):
                    if(self.image[px, py] == z):
                        N += 1
        return N
    
    def computeDensity(self):
        D = np.zeros(tuple([256 // PARAMS["activeContours"]["encodeColors"] for i in range(self.n)]))
        for px in range(self.image.shape[0]):
            for py in range(self.image.shape[1]):
                if(self.contour.interior[px, py]):
                    c = self.image[px, py] // PARAMS["activeContours"]["encodeColors"]
                    if(self.n == 1):
                        D[c] += 1
                    elif(self.n == 3):
                        D[c[0], c[1], c[2]] += 1
                    else:raise ValueError("Unvalid value for self.n = " + str(self.n))
        
        self.D = D.copy()
        self.lambd = 0.5 * np.min(self.D[self.D > 0])
        
    def getDensity(self, c):
        c = c // PARAMS["activeContours"]["encodeColors"]
        return self.D[c[0], c[1], c[2]]
    
    def p(self, z):
        return self.N(z) / self.area
    
    def H(self, Z, Q, lambd):
        return [self.N(Z[i])*Q[i] for i in range(len(Z))].sum() - lambd * self.area()
    
    def render(self, param):
        if(param == "contour"):
            self.contour.render()
        elif(param == "density"):
            if(self.n == 1):
                plt.imshow(self.D)
                plt.show()
            elif(self.n == 3):
                DD = np.zeros((16 * 256  // PARAMS["activeContours"]["encodeColors"]**2, 16 * 256 // PARAMS["activeContours"]["encodeColors"]**2))
                for i in range(256 // PARAMS["activeContours"]["encodeColors"]):
                    i1, i2 = i // 16 // PARAMS["activeContours"]["encodeColors"], i % 16 // PARAMS["activeContours"]["encodeColors"]
                    DD[256 * i1 // PARAMS["activeContours"]["encodeColors"]:256 * (i1 + 1) // PARAMS["activeContours"]["encodeColors"], 256 * i2 // PARAMS["activeContours"]["encodeColors"]:256 * (i2 + 1) // PARAMS["activeContours"]["encodeColors"]] += self.D[i].copy()
                plt.imshow(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(DD)))), cmap = "gray")
                plt.title(np.sum(DD))
                plt.show()
            else:raise ValueError("Unvalid value for self.n = " + str(self.n))
                
        else:raise ValueError("param unvalid for mask render, param = " + str(param))
        
class Windows():
    
    def __init__(self, image, n):
        self.image = mycv2.cvtBGRtoRGB(image)
        self.n = n
        # Création de la fenêtre principale (main window)
        Mywindows = tk.Tk()
        Mywindows.title('Image')
        
        # Création d'un widget Canvas (zone graphique)
        if(self.n == 1):
            self.height, self.width = image.shape
        else:
            self.height, self.width, _ = image.shape
            
        self.canvas = tk.Canvas(Mywindows, width = self.width, height = self.height, bg = "white")
        photo = ImageTk.PhotoImage(image=Image.fromarray(self.image))
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.pack(padx = 5, pady = 5)
        self.canvas.bind('<Button-1>', self.clic)
        self.corners = []
        self.clics = []
        self.lines = []
        tk.Button(Mywindows, text = 'delete', command = self.delete).pack(side=tk.LEFT, padx = 5, pady = 5)
        tk.Button(Mywindows, text = 'show hull', command = self.showHull).pack(side=tk.LEFT, padx = 5, pady = 5)
        tk.Button(Mywindows, text = 'create mask', command = self.computeMask).pack(side=tk.LEFT, padx = 5, pady = 5)
        Mywindows.mainloop()
        
    def delete(self):
        self.corners = []
        for i in range(len(self.clics)):
            self.canvas.delete(self.clics[i])
        self.clics = []
        for i in range(len(self.lines)):
            self.canvas.delete(self.lines[i])
        self.lines = []
        
    def computeMask(self):
        self.corners = np.array(self.corners)
        self.contour = Contour(np.array([self.corners[:, 1], self.corners[:, 0]]).transpose(), (self.height, self.width))
        self.mask = self.contour.interior
        
        if(debug):
            plt.figure(figsize = (15, 30))
            plt.subplot(131)
            plt.imshow(self.contour.array, cmap = "gray")
            plt.title("Hull")
            plt.subplot(132)
            plt.imshow(self.mask, cmap = "gray")
            plt.title("Mask")
            plt.subplot(133)
            if(self.n == 1):
                plt.imshow(self.image * self.mask, cmap = "gray")
            else:
                im = mycv2.cvtBGRtoRGB(self.image) * np.array([self.mask for i in range(self.n)]).transpose(1, 2, 0)
                plt.imshow(im.astype(np.float32), norm = None)
            plt.title("Masked image")
            plt.show()
        
    def showHull(self):
        for i in range(len(self.corners)):
            c1, c2 = self.corners[i], self.corners[(i + 1) % len(self.corners)]
            self.lines += [self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], width=2, fill='red')]
        
    def clic(self, event):
        """ Gestion de l'événement Clic gauche sur la zone graphique """
        # position du pointeur de la souris
        X = event.x
        Y = event.y
        self.corners += [[X, Y]]
        if(debug):
            print("Clic", X, Y)
        # on dessine un rond
        r = 5
        self.clics += [self.canvas.create_oval(X-r, Y-r, X+r, Y+r, outline='red', fill='red')]
        if(len(self.clics) > 1):
            c1, c2 = self.corners[-2], self.corners[-1]
            self.lines += [self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], width=2, fill='red')]