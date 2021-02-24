import numpy
import cv2
import math 

def rotation(img):
    th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 15)

    edges = cv2.Canny(th4, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180.0, 100, minLineLength=100, maxLineGap=5)

    angles = []
    for x1, y1, x2, y2 in lines[0]:
        angle = math.degrees(math.atan2(y2-y1, x2-x1))
        angles.append(angle)
    return angles