# Test contour/container detection
'''
import numpy as np
import cv2 as cv
# rgb_image = cv2.imread("black_container1.jpg")
rgb_image = cv2.imread("mtre4800-kawasaki-project/three_containers1.jpg")
# rgb_image = cv2.imread("shapes1.jpg")
rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
# img = cv.medianBlur(gray_image,5)
image = cv2.GaussianBlur(gray_image,(9,9), 4)

cv2.imshow("image0", image)
cv2.waitKey(0)


import sys
import cv2 as cv
import numpy as np
# def main(argv):
    
# src = cv2.imread("mtre4800-kawasaki-project/three_containers1.jpg")
src = cv2.imread("rgb_circle.png")
# rgb_image = cv2.imread("shapes1.jpg")
src = cv2.resize(src, (640, 480)) # (192, 224)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

gray = cv.medianBlur(gray, 5)

rows = gray.shape[0]
# circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
#                             param1=100, param2=30,
#                             minRadius=1, maxRadius=30)

circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,100,
                            param1=60,param2=40,minRadius=0,maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        print(i)
        # circle center
        cv.circle(src, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        cv.circle(src, center, radius, (255, 0, 255), 3)

cv.imshow("detected circles", src)
cv.waitKey(0)

# load the image and display it
rgb_image = cv2.imread("mtre4800-kawasaki-project/three_containers1.jpg")
# rgb_image = cv2.imread("shapes1.jpg")
rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)cv2.imshow("Image", image)
# convert the image to grayscale and threshold it
gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
### thresh1 = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
_, thresh2 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
_,thresh3 = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

# threshers = [thresh1, thresh2,thresh3]
# threshers = [thresh1, thresh2]
threshers = [thresh2, thresh3]
cv2.imshow("Thresh", np.hstack(threshers))
# cv2.imshow("Thresh", threshers)
cv2.waitKey(0)
'''

### Find contours and draw a bounding box

import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import seaborn as sn
import time
import cv2
import numpy as np
# import freenect

def contour_bounding_Rect(contour):
    # Regular bounding box
    # Contours -> rectangle box lengths
    x, y, w, h = cv2.boundingRect(contour)
    box = (x, y, w, h)
    return box

def contour_min_Area_Rect(contour):
    # Potentially rotated bounding box
    # Contours -> rectangle boundaries
    bounds = cv2.minAreaRect(contour)
#   min_box = (x, y, w, h)
    return bounds


# Choose camera
# cap = cv2.VideoCapture(0)

flag = True
while flag:
    # Set start time for FPS calculations
    # start_time = time.time()

    # Read from Windows laptop camera
    # ret, rgb_image = cap.read()

    # Read from Kinect camera
    # rgb_image, _ = freenect.sync_get_video()
    # rgb_image, _ = freenect.sync_get_depth()

    # rgb_image = rgb_image.astype(np.uint8)
    # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    # Read image from folder
    rgb_image = cv2.imread("mtre4800-kawasaki-project/three_containers4.jpg")     # 1-4
    # rgb_image = cv2.imread("mtre4800-kawasaki-project\one_black_container3.jpg")  # 1-3
    # rgb_image = cv2.imread("mtre4800-kawasaki-project/floor1.jpg")                # 1


    if rgb_image is None:
        # flag = False
        break

    # TODO: 1st Adjust these workspace/camera field of view boundaries
    # based on actual camera location
    # Zoom from top left and bottom right
    # rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)cv2.imshow("Image", image)
    # left_boundary = int(input("left_boundary: "))
    # right_boundary = int(input("right_boundary: "))
    # left_boundary = 0
    # right_boundary = 1000
    rgb_image = cv2.resize(rgb_image[300:3500,300:3500],(640, 480))
    # rgb_image = cv2.resize(rgb_image[0:1000,0:1000],(640, 480))
    # rgb_image = cv2.resize(rgb_image[0:510, 55:510],(640, 480))
    rgb_image = cv2.resize(rgb_image,(640, 480))

    # Grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)


    # TODO: ??? Adjust k_size threshold 
    # based on actual camera location
    # k_size = int(input("k_size:"))
    k_size = 5 # 0 - 8; 15
    kernelmatrix = np.ones((k_size, k_size), np.uint8)
    dilated = cv2.dilate(gray, kernelmatrix)


    # TODO: Look into using blurry (= More contours)
    # img = cv.medianBlur(gray_image,5)
    ###blurry = cv2.GaussianBlur(gray,(9,9), 1)
    blurry = cv2.GaussianBlur(dilated,(9,9), 2)
    # blurry = cv2.GaussianBlur(gray,(5,5), 1)
    # cv2.imshow('Blurry', blurry)
    # cv2.waitKey(0)

    # TODO: ??? Adjust mask threshold 
    # based on actual camera location
    # lower_range = (100, 0, 0)
    # upper_range = (120, 255, 255)
    # mask = cv2.inRange(gray, lower_range, upper_range)
    # lower_mask = int(input("lower_mask: "))
    # upper_mask = int(input("upper_mask: "))
    lower_mask = 40 #110 #100
    upper_mask = 130 #170 #155
    mask = cv2.inRange(blurry, lower_mask, upper_mask)
    # mask = cv2.inRange(blurry, 100, 155)
    # Black     *5-10,90            90,200          100,155
    # Orange    *70-75,110                  100,225 100,155
    # White     65-100,200-225      90,200  100,225 100,155

    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)

    # TODO: Refine these threshold values
    # Find Canny edges
    # edged = cv2.Canny(gray, 75, 200)
    # edged = cv2.Canny(mask, 100, 200)

    # Invert
    # TODO: ??? N/A Adjust invert threshold 
    # based on actual camera location
    # lower_invert = int(input("lower_invert: "))
    # upper_invert = int(input("upper_invert: "))
    lower_invert = 0 # any value (upper_invert-1)?
    upper_invert = 255
    _, inv_image = cv2.threshold(mask, lower_invert, upper_invert, cv2.THRESH_BINARY_INV)
    # cv2.imshow('Canny Edges', np.vstack([mask,edged]))
    cv2.imshow('Mask/Inv', np.vstack([mask,inv_image]))
    # cv2.imshow('Canny Edges', edged)
    cv2.waitKey(100)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image

    # Prev code
    '''
    [tR, tG, tB] = cv2.split(rgb_image)
    [iR, iG, iB] = cv2.split(rgb_image) #edged

    dR = cv2.absdiff(iR, tR)
    dG = cv2.absdiff(iG, tG)
    dB = cv2.absdiff(iB, tB)
    bl = 19
    cn = 15
    tR = cv2.adaptiveThreshold(dR,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bl,cn)
    tG = cv2.adaptiveThreshold(dG,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bl,cn)
    tB = cv2.adaptiveThreshold(dB,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bl,cn)
    difference = cv2.merge([tR,tG,tB])
    #cv2.imshow('fuzzy', cv2.merge([dR,dG,dB]))
    difference = cv2.cvtColor(difference,cv2.COLOR_BGR2GRAY)
    
    k_size = 15
    kernelmatrix = np.ones((k_size, k_size), np.uint8)
    d = cv2.dilate(difference, kernelmatrix)
    
    fuzzy = cv2.GaussianBlur(d, (9,9), 4)
    
    contours, _ = cv2.findContours(fuzzy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    '''

    contours, hierarchy = cv2.findContours(inv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))

    copy = rgb_image.copy()
    refined_contours = []
    contour_area_list = []

    # TODO: 4th Adjust contourArea threshold 
    # based on actual camera location
    
    # min_contour_area = int(input("min_contour_area:"))
    # max_contour_area = int(input("max_contour_area:"))
    min_contour_area = 4000
    max_contour_area = 3000000 #75000 #30000
    for c in contours:
        contour_area = cv2.contourArea(c,False)
        # contour_area = cv2.contourArea(c,True)
        print("contour_area:", contour_area)
        if abs(contour_area) >= min_contour_area and abs(contour_area) <= max_contour_area:
        # if contour_area >= 3000 and contour_area <= 50000:
            refined_contours.append(c)
            contour_area_list.append(contour_area)
        # epsilon = 0.1*cv2.arcLength(c,True)
        # epsilon = 0.1*cv2.arcLength(c,False)
        # print("epsilon", epsilon)
        # approx = cv2.approxPolyDP(c,epsilon,True)
        # approx_image = cv2.drawContours(copy, c, -1, (255,0,255), 5)
        # cv2.imshow("approx_image", approx_image)
        # cv2.waitKey(0)
    print("Number of Refined Contours found = " + str(len(refined_contours)))

    targets_boxes = []
    targets_bounds = []
    for c in refined_contours:
        boxes = contour_bounding_Rect(c)
        bounds = contour_min_Area_Rect(c)
        # cv2.approxPolyDP(c, approx, 5, True)
        targets_boxes.append(boxes)
        targets_bounds.append(bounds)
        # # epsilon = 0.1*cv2.arcLength(c,True)
        # epsilon = 0.1*cv2.arcLength(c,False)
        # print("epsilon", epsilon)
        # approx = cv2.approxPolyDP(c,epsilon,True)
        # approx_image = cv2.drawContours(copy, approx, -1, (255,0,255), 5)
        # cv2.imshow("approx_image", approx_image)
        # cv2.waitKey(5000)

    # boxes = [contour_box(c) for c in new_contours]

    copy0 = rgb_image.copy()
    contours_image = cv2.drawContours(copy0, contours, -1, (255,0,255), 2)
    # cv2.imshow('Contours Image', contours_image)
    # cv2.waitKey(0)

    copy1 = rgb_image.copy()
    new_contours_image = cv2.drawContours(copy1, refined_contours, -1, (255,0,255), 2)
    # cv2.imshow('New Contours Image', new_contours_image)
    # cv2.waitKey(0)

    final = rgb_image.copy()
    for index,boxes in enumerate(targets_boxes):
        x1,x2,y1,y2 = boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]
        box_center = (int((x1+y1)/2),int((x2+y2)/2))
        cv2.rectangle(final, (x1, x2), (y1, y2), (0, 255, 0), 2)
        cv2.putText(final, f'{int(contour_area_list[index])}',box_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(final, box_center, (int(final.shape[1]/2),int(final.shape[0]/2)), (100,0,100), 2)

    for bounds in targets_bounds:
        # Rectangle boundaries -> Box
        box = cv2.boxPoints(bounds)
        # Box corners -> int
        box = np.int0(box)
        # x1,x2,y1,y2 = bounds[0], bounds[1], bounds[0] + bounds[2], bounds[1] + bounds[3]
        # Display regular bounding box
        # cv2.rectangle(copy2, (x1, x2), (y1, y2), (255, 0, 0), 2)
        # cv2.line(final, (int((box[0][0]+box[1][0])/2),int((box[1][1]+box[2][1])/2)), (int(final.shape[1]/2),int(final.shape[0]/2)), (100,0,100), 2)
        cv2.drawContours(final, [box], 0, (255, 255, 0), 2)

    
    # cv2.imshow('Boxes', copy)
    cv2.imshow('Boxes', final)
    # cv2.waitKey(0)
    cv2.waitKey(1000)

    # Draw and display all contours
    # -1 signifies drawing all contours
    # cv2.drawContours(rgb_image, contours, -1, (255, 0, 0), 3)
    # cv2.imshow('Contours', rgb_image)
    # cv2.waitKey(0)

    # flag: Used to stop while loop after 1 run/pass
    # flag = False
    if (cv2.waitKey(1) & 0xFF) == 27:
        break

# cap.release()
cv2.destroyAllWindows()

