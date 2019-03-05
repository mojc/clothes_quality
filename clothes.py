# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:33:34 2017

@author: Mojca
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
#255 is black
#%%
img_orig = cv2.imread('orig.jpg')
img_cut = cv2.imread('cut.jpg')
img_stain = cv2.imread('stain.jpg')

#%% color to gray
img_orig_G = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
img_cut_G = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
img_stain_G = cv2.cvtColor(img_stain, cv2.COLOR_BGR2GRAY)

#%% histogram
#plt.hist(img_stain_G.ravel(),256,[0,256]); plt.show()

#%% threshold -->> automatical thresholding 
ret,thresh_orig = cv2.threshold(img_orig_G, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh_cut = cv2.threshold(img_cut_G, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,thresh_stain = cv2.threshold(img_stain_G, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#%% tample matching
#orig_template = cv2.matchTemplate(thresh_orig, thresh_orig, cv2.TM_CCORR_NORMED)
#stain_template = cv2.matchTemplate(thresh_stain, thresh_orig, cv2.TM_CCORR_NORMED)
#cut_template = cv2.matchTemplate(thresh_cut, thresh_orig, cv2.TM_CCORR_NORMED)
#%%difference
#important orrder!! what if white t-shirt on black
if thresh_orig[0,0] == 255:
    stain = thresh_stain - thresh_orig
    cut = thresh_cut - thresh_orig 
elif thresh_orig[0,0] == 0:
    stain =  thresh_orig - thresh_stain
    cut =  thresh_orig - thresh_cut
#%%
kernel = np.ones((25,25),np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(25,25))
close_cut = cv2.morphologyEx(cut, cv2.MORPH_OPEN, kernel)
close_stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, kernel) 

#%%difference between model and test -> number and %
def white(img):
    white = 0
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] == 0:
                white = white +1
    procent = (white/(img.shape[0]*img.shape[1]))*100
    return white
#%% messures of similarity!!!
ret,thresh_orig_w = cv2.threshold(img_orig_G, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#%%
orig_white = white(thresh_orig_w)
stain_error = white(stain)
cut_error = white(cut)
stain_error_proc = stain_error/orig_white
cut_error_proc = cut_error/orig_white

print("Erros proc:", stain_error_proc)
print("Erros proc:", cut_error_proc)
#%%
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', thresh_stain)
#%%
cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.imshow('image2', close_stain)
#%%
cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
cv2.imshow('image3', stain_E)
cv2.waitKey(0)
cv2.destroyAllWindows()