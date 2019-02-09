# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:10:48 2018

@author: juduq

Distribution de Python : Anaconda
Python  3.6.3
"""
import copy
import numpy as np
import scipy as sc
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import *

image = sc.ndimage.imread('image.png',flatten=True)#Importing image in black and white
image_colored = sc.ndimage.imread('image.png',mode = 'RGB')#Also importing in colors, that will be useful for final graphics

filtre = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])#Sobel filter

def filt2D(sign,noyau):
    return(sc.signal.convolve2d(sign,noyau,mode='same'))
    
def sobel(img):
    return filt2D(img,filtre),filt2D(img,filtre.T)#Implementing Sobel filter in both directions

def gaussien(sigma):#Definition of response to gaussian filter
    dim = int(2*sigma+1)
    h = np.zeros((dim,dim))
    for k in range(dim):
        for l in range(dim):
            h[k][l]=np.exp(-(k**2+l**2)/sigma**2)
    Z = sum(h)
    h= h*(1/Z)
    return h

def lissage(img,largeur):
    return(filt2D(img,gaussien(largeur)))

#Edge detection
def normalisation(matrice):#This function allow us to append lines and columns of zeros around a matrix
        matrice_elargie = np.concatenate((matrice,np.zeros((1,matrice.shape[1]))))
        matrice_elargie = np.concatenate((matrice_elargie,np.zeros((matrice_elargie.shape[0],1))),axis = 1)
        matrice_elargie = np.concatenate((np.zeros((matrice_elargie.shape[0],1)),matrice_elargie),axis = 1)
        matrice_elargie = np.concatenate((np.zeros((1,matrice_elargie.shape[1])),matrice_elargie))
        return(matrice_elargie)
        
def thinning(norme,direction):#We keep gradient's maximas by calling gradient_discrete to discretize the gradient
    contours = np.zeros((norme.shape[0],norme.shape[1]))
    norme_elargie = normalisation(norme)
    for x in range(norme.shape[0]):
        for y in range(norme.shape[1]): 
            x_grad,y_grad = gradient_discrete(direction[x][y])
            if(norme[x][y]>=norme_elargie[x+1+x_grad][y+1+y_grad] and norme[x][y]>=norme_elargie[x+1-x_grad][y+1-y_grad]):
                contours[x][y] = norme[x][y]    
    return(contours)
            
def gradient_discrete(theta):
    if(theta<=3*np.pi/8 and theta >= -3*np.pi/8):
        x = 1
    elif( (theta >= 3*np.pi/8 and theta<=5*np.pi/8) or ((theta <= -3*np.pi/8 and theta>=-5*np.pi/8))):
        x = 0
    else : 
        x = -1
    if(theta >= np.pi/8 and theta <= 7*np.pi/8):
        y= 1
    elif(theta <= -np.pi/8 and theta>=-7*np.pi/8):
        y=-1
    else:
        y=0
    return(int(x),int(y))
    
def thresholding(contours,seuilinf,seuilsup):#Isolate edges
    contours_forts= 255*(contours>seuilsup).astype(np.int)
    contours_faibles = 255*(np.logical_and(contours<seuilsup,contours>seuilinf)).astype(np.int)
    return(contours_forts,contours_faibles)

#Coursing over the graph of edges

def hysteresis(contoursfaibles,contoursforts):
    '''This function gather weak and strong edges and return connex components'''
    contours_all = normalisation(contoursfaibles + contoursforts)
    compconnexes=[]
    indices_contours_forts = []
    tab_indices_contours_forts = np.where(contoursforts>0)
    for i in range(len(tab_indices_contours_forts[0])):#We build a list of strong edges
        indices_contours_forts.append((tab_indices_contours_forts[0][i]+1,tab_indices_contours_forts[1][i]+1))
    while(len(indices_contours_forts)>0):#While the list isn't empty
        comp = []
        stack = [(indices_contours_forts[0][0],indices_contours_forts[0][1])]#We begin from the next strong edge we append to the stack
        while(len(stack)>0):#And we course the adjacent graph 
            cur=stack.pop()
            comp.append(cur)
            if(cur in indices_contours_forts):
                indices_contours_forts.remove(cur)
            sommets_adj = [(i,j) for j in range(cur[1]-1,cur[1]+2) for i in range(cur[0]-1,cur[0]+2)]
            for node in sommets_adj:
                if (node not in comp) and contours_all[node]>0:
                    stack.append(node)#We add to the stack all the adjacent pixels
        compconnexes.append(comp)#Comp is a list of all fetched pixels
    contours_final = np.zeros(contours_faibles.shape)
    for com in compconnexes:
        for i in com:
            contours_final[i[0]-1][i[1]-1] = 255 #We fixe the value of fetched pixels to 255
    return contours_final

#Graph segmentation
def get_children(graphe,indice,i):
    resultat = []
    x= indice[0]
    y=indice[1]
    if(graphe[x+1][y]==0):
        graphe[x+1,y]=i
        resultat.append((x+1,y))
    if(graphe[x,y+1]==0):
        graphe[x,y+1]=i
        resultat.append((x,y+1))
    if(graphe[x-1,y]==0):
        graphe[x-1,y]=i
        resultat.append((x-1,y))
    if(graphe[x,y-1]==0):
        graphe[x,y-1]=i
        resultat.append((x,y-1))
    return resultat
    
def segmentation(contours):#Segmenting the graph into different connex components
    graphe = copy.deepcopy(contours)    
    graphe[0,:]=255
    graphe[:,0]=255
    graphe[graphe.shape[0]-1,:]=255
    graphe[:,graphe.shape[1]-1] = 255
    i=1
    zeros = np.where(graphe==0)
    while(len(zeros[0])>0):
        stack = [(zeros[0][0],zeros[1][0])]#We course the graph again with the same idea
        graphe[(zeros[0][0],zeros[1][0])]=i
        while(len(stack)>0):
            cur = stack.pop()
            for j in get_children(graphe,cur,i):#Get_children get the adjacent vertices non edges
                stack.append(j)
        i+=1
        zeros = np.where(graphe==0)
    return graphe

#Plotting results
def color(x):
    if(x>len(tab_colors)-1):
        return [0,0,0]
    else:
        return tab_colors[int(x)]

lisse = lissage(image,1)
resultat = sobel(lisse)

norme = np.sqrt(resultat[0]**2+resultat[1]**2)
direction = np.arctan2(resultat[0],resultat[1])
contours = thinning(norme,direction)
contours_forts,contours_faibles = thresholding(contours,120,640)#sigma = 1 :120,640

contours_all = hysteresis(contours_faibles,contours_forts)

comp_connexes = segmentation(contours_all)
fig = plt.figure(figsize=(24,36))

ax_orig=fig.add_subplot(2,3,1)
ax_orig.imshow(image.astype(int),cmap = 'gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_aminci=fig.add_subplot(2,3,2)
ax_aminci.imshow(contours.astype(int),cmap='gray')
ax_aminci.set_title('Gradient thinned')
ax_aminci.set_axis_off()

ax_lisse =fig.add_subplot(2,3,3)
ax_lisse.imshow(lisse.astype(int),cmap = 'gray')
ax_lisse.set_title('Smoothing')
ax_lisse.set_axis_off()

ax_norme=fig.add_subplot(2,3,4)
ax_norme.imshow(norme.astype(int),cmap='gray')
ax_norme.set_title('Norm of gradient')
ax_norme.set_axis_off()


image_colored[:,:,0][np.where(contours_all>0)] = 255
image_colored[:,:,1:3][np.where(contours_all>0)] = 0
tab_colors=[]
image_rainbow = np.zeros(image_colored.shape)
n=int(np.max(comp_connexes))
for i in range(n):
    tab_colors.append([randint(0,256),randint(0,256),randint(0,256)])
for i in range(0,comp_connexes.shape[0]):
    for j in range(0,comp_connexes.shape[1]):
        image_rainbow[i][j] = color(comp_connexes[i][j])

ax_rainbow =fig.add_subplot(2,3,5)
ax_rainbow.imshow(image_rainbow.astype(int))
ax_rainbow.set_title('Connex components')
ax_rainbow.set_axis_off()

ax_final =fig.add_subplot(2,3,6)
ax_final.imshow(image_colored.astype(int))
ax_final.set_title('Final edges')
ax_final.set_axis_off()

