import cv2
import numpy as np  
import matplotlib.pyplot as plt  
from line_segmentation import line_segmentation, createFolder
import os
#import glob
import re

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def words(img, minArea=0):
    _, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (9, 15))
    dil = cv2.dilate(th1, kernel, iterations=3)    
    contours, hier = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = []
    for c in contours:
        if cv2.contourArea(c) < minArea:
            continue
        currBox = cv2.boundingRect(c)
        (x, y, w, h) = currBox
        currImg = img[y:y+h, x:x+w]
        res.append((currBox, currImg))
    #contuors.sort(key=lambda x:get_contour_precedence(x, contour.shape[1]))
    return sorted(res, key=lambda entry:entry[0][0])
    
def word_segmentation(imgFiles):
    imgFiles = os.listdir('../line segment images/')
    line_segs = natural_sort(imgFiles)
    for (i,f) in enumerate(line_segs):
        print('Segmenting words of %s'%f)
        img = cv2.imread('../line segment images/%s'%f, 0)
        res = words(img, minArea=2000)
        if not os.path.exists('../word segment images/%s'%f):
            createFolder('../word segment images/%s'%f)
            #os.mkdir('../word segment images/%s'%f)
            print('Segmented into %d words'%len(res))
        for (j, w) in enumerate(res):
            (wordBox, wordImg) = w
            (x, y, w, h) = wordBox
            cv2.imwrite('../word segment images/%s/%d.jpg'%(f, j), wordImg) # save word
            cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
        
        # output summary image with bounding boxes around words   
        cv2.imwrite('../word segment images/%s/summary.jpg'%f, img)
    