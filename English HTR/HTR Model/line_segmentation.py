import cv2
import os
import numpy as np  
from scipy import ndimage
from imgprocess import img_preprocess
#from rotation import rotation
from resize_image import resize_image
from prespective import prespective_transform

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        pass

def get_contour_precedence(contour, cols):
    tolerance_factor = 150
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

#img = cv2.imread("C:/Users/luqman/Desktop/OCR Handwritten Recognition/Scanned doc text detection and segmentation/line segmentation/1.jpg")

def line_segmentation(image):
#prespective transformation
    img_pres = prespective_transform(image)
    img_gray = cv2.cvtColor(img_pres, cv2.COLOR_BGR2GRAY)

#image resize
    resize = resize_image(img_gray)

#image dilation and rotated too
    img_dil = img_preprocess(img_gray)

#rotation angle
#img_angles = rotation(img_gray)
#median_angle = np.median(img_angles)

#rotate resize image
#img_rotated = ndimage.rotate(resize, median_angle)
#dil_rotated = ndimage.rotate(img_dil, median_angle)

#find contours
#contours, hierarchy = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = cv2.findContours(img_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#sort contours
    contours.sort(key=lambda x:get_contour_precedence(x, img_dil.shape[1]))
        
    for i, ctr in enumerate(contours):
        mult = 0.95
        rect = cv2.minAreaRect(ctr)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        W = rect[1][0]
        H = rect[1][1]
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        rotated = False
        angle = rect[2]
        if angle < -45:
            angle+=90
            rotated = True
        center = (int((x1+x2)/2), int((y1+y2)/2))
        size = (int(mult*(x2-x1)),int(mult*(y2-y1)))
        M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1)
        cropped = cv2.getRectSubPix(resize, size, center)    
        cropped = cv2.warpAffine(cropped, M, (2500, 140), borderValue=200)

        if hierarchy[0][i][2] == -1 :
            if  cv2.contourArea(ctr) > 10000:
                folder = createFolder('../line segment images/')
                cv2.imwrite(os.path.join('../line segment images/', 'line_'+str(i)+'.jpg'), cropped)
                #cv2.imwrite("C:/Users/luqman/Desktop/OCR Handwritten Recognition/Scanned doc text detection and segmentation/segmentation/line segment images/line_" +str(i)+ ".jpg", cropped)
    
    for i in range(len(contours)):
        cv2.drawContours(resize, contours, -1, (0, 255, 0), 1, cv2.LINE_AA)
        img = cv2.putText(resize, str(i), cv2.boundingRect(contours[i])[:2], cv2.FONT_HERSHEY_TRIPLEX, 2, [125])
        cv2.imwrite('../sorted_lines.jpg', img)
        #cv2.imwrite('C:/Users/luqman/Desktop/OCR Handwritten Recognition/Scanned doc text detection and segmentation/segmentation/sorted img.jpg', img)
    

