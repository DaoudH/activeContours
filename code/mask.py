# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 19:16:07 2018

@author: romai
"""

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import mycv2 as mycv2
import matplotlib.pyplot as plt
import yaml
verbose = yaml.load(open("params.yaml"))["verbose"]

class Mask:
    
    def __init__(self, image):
        if(len(image.shape) == 2):
            self.n = 1
        else:
            self.n = image.shape[-1]
        self.image = image
        self.hull, self.mask = self.initializeMask()
        
    def initializeMask(self):
        windows = Windows(self.image, self.n)
        return windows.hull, windows.mask
    
    def area(self):
        return self.mask.sum() / (self.mask.shape[0] * self.mask.shape[0])
    
    def N(self, z):
        N = 0
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if(self.mask[i, j]):
                    if(self.image[i, j] == z):
                        N += 1
        return N
    
    def p(self, z):
        return self.N(z) / self.area
    
    def H(self, Z, Q, lambd):
        return [self.N(Z[i])*Q[i] for i in range(len(Z))].sum() - lambd * self.area()
    
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
        hull = np.zeros((self.width, self.height))
            
        for c in range(len(self.corners)):
            c1, c2 = self.corners[c], self.corners[(c + 1) % len(self.corners)]
            for alpha in np.linspace(0, 1, 2*max(self.width, self.height)):
                hull[int(c1[0] + alpha * (c2[0] - c1[0])), int(c1[1] + alpha * (c2[1] - c1[1]))] = 1
        
        mask = np.zeros((self.width, self.height))
        
        for px in range(self.width):
            cur = hull[px, 0]
            toAdd = np.zeros((self.width, self.height))
            for py in range(self.height):
                if(hull[px, py] == 1 and py != 0 and hull[px, py - 1] == 0):
                    cur = (cur + 1) % 2
                toAdd[px, py] = cur
                
            if(cur != 1):
                mask += toAdd
                
        mask += hull
        self.mask = np.minimum(mask, 1).transpose()
        self.hull = hull.transpose()
        
        if(verbose):
            plt.figure(figsize = (10, 30))
            plt.subplot(131)
            plt.imshow(self.hull, cmap = "gray")
            plt.title("Hull")
            plt.subplot(132)
            plt.imshow(self.mask, cmap = "gray")
            plt.title("Mask")
            plt.subplot(133)
            if(self.n == 1):
                plt.imshow(self.image * self.mask, cmap = "gray")
            else:
                plt.imshow(self.image * np.array([self.mask for i in range(self.n)]).transpose(1, 2, 0))
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
        if(verbose):
            print("Clic", X, Y)
        # on dessine un rond
        r = 5
        self.clics += [self.canvas.create_oval(X-r, Y-r, X+r, Y+r, outline='red', fill='red')]
        if(len(self.clics) > 1):
            c1, c2 = self.corners[-2], self.corners[-1]
            self.lines += [self.canvas.create_line(c1[0], c1[1], c2[0], c2[1], width=2, fill='red')]