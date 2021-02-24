import numpy as np  
import cv2
from scipy import ndimage
from resize_image import resize_image
#from rotation import rotation

def img_preprocess(filename):
    #img = cv2.imread(filename)
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    resize = resize_image(filename)
    blur = cv2.blur(resize, (5, 5))

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    _, th2 = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((1, 1),np.uint8)
    dst = cv2.filter2D(th2,-1,kernel)
    laplacian = cv2.Laplacian(dst, cv2.CV_8U)

    kernel1 = np.ones((5,5),np.uint8)/20
    dst1 = cv2.filter2D(laplacian,-1,kernel1)
    
    #kernel2 = np.ones((5,5),np.uint8)/20
    dst2 = cv2.filter2D(dst1,-1,kernel1)

    #kernel3 = np.ones((5,5), np.uint8)/20
    dst3 = cv2.filter2D(dst2,-1,kernel1)

    _, th1 = cv2.threshold(dst3, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    kernel4 = np.ones((10,10), np.uint8)
    img_dil1 = cv2.dilate(th1, kernel4, iterations=3)
    img_f = cv2.bitwise_and(img_dil1, th3)
    med = cv2.medianBlur(img_f, 5)

    kernel5 = np.ones((5, 5), np.uint8)
    fl = cv2.morphologyEx(med, cv2.MORPH_CLOSE, kernel5)

    bit1 = cv2.bitwise_and(med, fl)
    #img_x = resize_image(bit1)

    #img_angles = rotation(filename)
    #median_angle = np.median(img_angles)
    #img_dil = ndimage.rotate(bit1, median_angle)

    kernel6 = np.ones((5,100), np.uint8)
    img_dilation = cv2.dilate(bit1, kernel6, iterations=2)

    #cv2.imwrite('word segmentation/img_dilation.jpg', img_dilation)
    return img_dilation
