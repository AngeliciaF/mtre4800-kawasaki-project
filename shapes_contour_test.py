"""
Contours shape recognition mainly based on cv2.approxPolyDP() function
"""

# # Import required packages:
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

'''
def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """Gives the coordinates to draw centered"""

    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)


def detect_shape(contour):
    """Returns the shape (e.g. 'triangle', 'square') from the contour"""

    detected_shape = '-----'

    # Calculate perimeter of the contour:
    perimeter = cv2.arcLength(contour, True)

    # Get a contour approximation:
    contour_approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)

    # Check if the number of vertices is 3. In this case, the contour is a triangle
    if len(contour_approx) == 3:
        detected_shape = 'triangle'

    # Check if the number of vertices is 4. In this case, the contour is a square/rectangle
    elif len(contour_approx) == 4:

        # We calculate the aspect ration from the bounding rect:
        x, y, width, height = cv2.boundingRect(contour_approx)
        aspect_ratio = float(width) / height

        # A square has an aspect ratio close to 1 (comparison chaining is used):
        if 0.90 < aspect_ratio < 1.10:
            detected_shape = "square"
        else:
            detected_shape = "rectangle"

    # Check if the number of vertices is 5. In this case, the contour is a pentagon
    elif len(contour_approx) == 5:
        detected_shape = "pentagon"

    # Check if the number of vertices is 6. In this case, the contour is a hexagon
    elif len(contour_approx) == 6:
        detected_shape = "hexagon"

    # The shape as more than 6 vertices. In this example, we assume that is a circle
    else:
        detected_shape = "circle"

    # return the name of the shape and the found vertices
    return detected_shape, contour_approx


def array_to_tuple(arr):
    """Converts array to tuple"""

    return tuple(arr.reshape(1, -1)[0])


def draw_contour_points(img, cnts, color):
    """Draw all points from a list of contours"""

    for cnt in cnts:
        print(cnt.shape)
        squeeze = np.squeeze(cnt)
        print(squeeze.shape)

        for p in squeeze:
            pp = array_to_tuple(p)
            cv2.circle(img, pp, 10, color, -1)

    return img


def draw_contour_outline(img, cnts, color, thickness=1):
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 9))
plt.suptitle("Shape recognition based on cv2.approxPolyDP()", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
# image = build_sample_image_2()
# rgb_image = cv2.imread("black_container1.jpg")
rgb_image = cv2.imread("shapes1.jpg")
rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a binary image:
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# Find contours using the thresholded image:
# Note: cv2.findContours() has been changed to return only the contours and the hierarchy
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Show the number of detected contours:
print("detected contours: '{}' ".format(len(contours)))

# Make a copy to draw the results:
image_contours = rgb_image.copy()
image_recognition_shapes = rgb_image.copy()

# Draw the outline of all detected contours:
draw_contour_outline(image_contours, contours, (255, 255, 255), 4)

for contour in contours:
    # Compute the moments of the current contour:
    M = cv2.moments(contour)

    # Calculate the centroid of the contour from the moments:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # Detect shape of the current contour:
    shape, vertices = detect_shape(contour)

    # Draw the detected vertices:
    draw_contour_points(image_contours, [vertices], (255, 255, 255))

    # Get the position to draw:
    (x, y) = get_position_to_draw(shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)

    # Write the name of shape on the center of shapes
    cv2.putText(image_recognition_shapes, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)

# Plot the images
show_img_with_matplotlib(rgb_image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)
show_img_with_matplotlib(image_contours, "contours outline (after approximation)", 3)
show_img_with_matplotlib(image_recognition_shapes, "contours recognition", 4)

# Show the Figure:
plt.show()
'''

# import numpy as np
# import cv2 as cv
# # rgb_image = cv2.imread("black_container1.jpg")
# rgb_image = cv2.imread("mtre4800-kawasaki-project/three_containers1.jpg")
# # rgb_image = cv2.imread("shapes1.jpg")
# rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)
# gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
# # img = cv.medianBlur(gray_image,5)
# image = cv2.GaussianBlur(gray_image,(9,9), 4)

# cv2.imshow("image0", image)
# cv2.waitKey(0)


# import sys
# import cv2 as cv
# import numpy as np
# # def main(argv):
    
# # src = cv2.imread("mtre4800-kawasaki-project/three_containers1.jpg")
# src = cv2.imread("rgb_circle.png")
# # rgb_image = cv2.imread("shapes1.jpg")
# src = cv2.resize(src, (640, 480)) # (192, 224)
# gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# gray = cv.medianBlur(gray, 5)

# rows = gray.shape[0]
# # circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
# #                             param1=100, param2=30,
# #                             minRadius=1, maxRadius=30)

# circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,100,
#                             param1=60,param2=40,minRadius=0,maxRadius=0)

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         center = (i[0], i[1])
#         print(i)
#         # circle center
#         cv.circle(src, center, 1, (0, 100, 100), 3)
#         # circle outline
#         radius = i[2]
#         cv.circle(src, center, radius, (255, 0, 255), 3)

# cv.imshow("detected circles", src)
# cv.waitKey(0)

# # load the image and display it
# rgb_image = cv2.imread("mtre4800-kawasaki-project/three_containers1.jpg")
# # rgb_image = cv2.imread("shapes1.jpg")
# rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)cv2.imshow("Image", image)
# # convert the image to grayscale and threshold it
# gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
# ### thresh1 = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
# _, thresh2 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
# _,thresh3 = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)

# # threshers = [thresh1, thresh2,thresh3]
# # threshers = [thresh1, thresh2]
# threshers = [thresh2, thresh3]
# cv2.imshow("Thresh", np.hstack(threshers))
# # cv2.imshow("Thresh", threshers)
# cv2.waitKey(0)

### Find contours and draw a bounding box

from ast import match_case
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import seaborn as sn
import time
import cv2
import numpy as np


def contour_bounding_Rect(contour):
    """Return a bounding box for a given contour."""
    x, y, w, h = cv2.boundingRect(contour)
    box = (x, y, w, h)
    return box

def contour_min_Area_Rect(contour):
    """Return a bounding box for a given contour."""
    bounds = cv2.minAreaRect(contour)
#   min_box = (x, y, w, h)
    return bounds


# Choose camera
# cap = cv2.VideoCapture(0)
flag = True
while flag:
    # Set start time for FPS calculations
    start_time = time.time()

    # Read from camera
    # ret, rgb_image = cap.read()

    # Let's load a simple image with 3 black squares
    rgb_image = cv2.imread("mtre4800-kawasaki-project/three_containers4.jpg")
    rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)cv2.imshow("Image", image)
    
    if rgb_image is None:
        break

    # Grayscale
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # TODO: Look into using blurry (= More contours)
    # img = cv.medianBlur(gray_image,5)
    ###blurry = cv2.GaussianBlur(gray,(9,9), 1)
    blurry = cv2.GaussianBlur(gray,(9,9), 2)
    # blurry = cv2.GaussianBlur(gray,(5,5), 1)
    # cv2.imshow('Blurry', blurry)
    # cv2.waitKey(0)

    # TODO: Look into using mask (= More contours)
    lower_range = (100, 0, 0)
    upper_range = (120, 255, 255)
    # mask = cv2.inRange(gray, lower_range, upper_range)
    lower = int(input("Lower: "))
    upper = int(input("Upper: "))
    mask = cv2.inRange(blurry, lower, upper)
    # mask = cv2.inRange(blurry, 100, 155)
    # Black     *5-10,90            90,200          100,155
    # Orange    *70-75,110                  100,225 100,155
    # White     65-100,200-225      90,200  100,225 100,155

    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)

    # TODO: Refine these threshold values
    # Find Canny edges
    # edged = cv2.Canny(gray, 75, 200)
    edged = cv2.Canny(mask, 100, 200)
    _, inv_image = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('Canny Edges', np.vstack([mask,edged]))
    cv2.imshow('Mask/Canny Edges', np.vstack([mask,inv_image]))
    # cv2.imshow('Canny Edges', edged)
    cv2.waitKey(1000)

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

    # TODO: Adjust contourArea threshold
    new_contours = [c for c in contours if cv2.contourArea(c) >= 5000] # and cv2.contourArea(c) <= 2000]
    print("Number of New Contours found = " + str(len(new_contours)))

    targets_boxes = []
    targets_bounds = []
    for c in new_contours:
        boxes = contour_bounding_Rect(c)
        bounds = contour_min_Area_Rect(c)
        # cv2.approxPolyDP(c, approx, 5, True)
        targets_boxes.append(boxes)
        targets_bounds.append(bounds)
    # boxes = [contour_box(c) for c in new_contours]

    copy = rgb_image.copy()
    contours_image = cv2.drawContours(copy, contours, -1, (255,0,255), 2)
    # cv2.imshow('Contours Image', contours_image)
    # cv2.waitKey(0)

    copy1 = rgb_image.copy()
    new_contours_image = cv2.drawContours(copy1, new_contours, -1, (255,0,255), 2)
    # cv2.imshow('New Contours Image', new_contours_image)
    # cv2.waitKey(0)

    copy2 = rgb_image.copy()
    for boxes in targets_boxes:
        # Contours -> rectangle boundaries
        # bounds = cv2.minAreaRect(boxes)
        # # Rectangle boundaries -> Box
        # box = cv2.boxPoints(bounds)
        # # Box corners -> int
        # box = np.int0(box)
        x1,x2,y1,y2 = boxes[0], boxes[1], boxes[0] + boxes[2], boxes[1] + boxes[3]
        cv2.rectangle(copy2, (x1, x2), (y1, y2), (0, 255, 0), 2)

    for bounds in targets_bounds:
        # Contours -> rectangle boundaries
        # bounds = cv2.minAreaRect(bounds)
        # Rectangle boundaries -> Box
        box = cv2.boxPoints(bounds)
        # Box corners -> int
        box = np.int0(box)
        # x1,x2,y1,y2 = bounds[0], bounds[1], bounds[0] + bounds[2], bounds[1] + bounds[3]
        # Display regular bounding box
        cv2.rectangle(copy2, (x1, x2), (y1, y2), (255, 0, 0), 2)
        cv2.drawContours(copy2, [box], 0, (255, 255, 0), 2)

    
    # cv2.imshow('Boxes', copy)
    cv2.imshow('Boxes', copy2)
    # cv2.waitKey(0)

    # Draw and display all contours
    # -1 signifies drawing all contours
    # cv2.drawContours(rgb_image, contours, -1, (255, 0, 0), 3)
    # cv2.imshow('Contours', rgb_image)
    # cv2.waitKey(0)

    # flag = False
    if (cv2.waitKey(1) & 0xFF) == 27:
        break

# cap.release()
cv2.destroyAllWindows()


# Plot rotating bounding boxes
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
DPI = 72

# Rotation centre. It is helpful to have this with shape (2,1)
cx, cy = 150, 50
rc = np.array(((cx, cy),)).T

# Initial width and height of the bounding rectangle we will fit the object in.
rw, rh = 200, 300
# Initial corner positions of the bounding rectangle 
x1, y1, x2, y2 = 250, 50, 250+rw, 50+rh

def rotate_points(pts, theta, rc):
    """Rotate the (x,y) points pts by angle theta about centre rc."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    return rc + R @ (pts - rc)


def plot_poly(pts, colour='tab:blue', lw=2, opacity=1, ls='-'):
    """Plot a closed polygon with vertices at pts."""

    plot_pts = np.vstack((pts.T, pts[:,0]))
    ax.plot(*zip(*plot_pts), c=colour, lw=lw, alpha=opacity, ls=ls)


def plot_obj(pts, colour='tab:green', lw=2):
    """Draw the object we are rotating: a circle and polygon."""

    plot_poly(pts[:,1:], colour, lw=lw, opacity=0.5)
    circle = Circle(pts[:,0], obj_cr, edgecolor=colour, fill=False, lw=lw,
                    alpha=0.5)
    ax.add_patch(circle)


def get_boundary_pts(pts):
    """Get the vertices of the bounding rectangle for the points pts."""

    xmin, xmax = np.min(pts[0]), np.max(pts[0])
    ymin, ymax = np.min(pts[1]), np.max(pts[1])
    return np.array(((xmin,ymin), (xmax,ymin), (xmax, ymax), (xmin, ymax))).T


def get_obj_boundary(obj_pts):
    """Get the vertices of the bounding rectangle for the rotated object."""

    fcx, fcy = obj_pts[:,0]
    # Get the boundary from the triangle coordinates and the circle limits
    _obj_boundary = np.vstack((obj_pts.T[1:], (fcx-obj_cr, fcy),
                (fcx+obj_cr, fcy), (fcx, fcy-obj_cr), (fcx, fcy+obj_cr))).T
    return get_boundary_pts(_obj_boundary)




fig, ax = plt.subplots(figsize=(8.33333333, 8.33333333), dpi=DPI)

# Initial bounding rectangle of unrotated object.
pts = np.array( ((x1,y1), (x2,y1), (x2,y2), (x1,y2)) ).T
# The radius of the circle in our plotted object.
obj_cr = (rh - rw*np.sqrt(3)/2)/2
# The coordinates defining our object.
obj_pts = np.array( ((x1 + rw/2, y1 + rh - obj_cr),    # circle centre
                     (x1, y1), (x2, y1),               #
                     (x1+rw/2, y2-2*obj_cr),           # triangle
                    )).T
# Plot the unrotated object and its bounding rectangle
# plot_obj(obj_pts)
# plot_poly(pts)

nrots = 60
theta = np.radians(360 // nrots)
boundary_trail_pts, obj_boundary_trail_pts = [], []
for i in range(nrots):
    fig, ax = plt.subplots(figsize=(8.33333333, 6.25), dpi=DPI)
    ax.set_xlim(-600,600)
    ax.set_ylim(-600,600)
    # Indicate the centre of rotation
    ax.add_patch(Circle((cx,cy), 10))

    # Plot the object
    plot_obj(obj_pts)
    # Plot the rotated object's boundary
    # boundary_pts = get_obj_boundary(obj_pts)
    # plot_poly(boundary_pts, colour='tab:purple', ls='--')
    # obj_boundary_trail_pts.append(np.mean(boundary_pts, axis=1))
    # ax.plot(*zip(*obj_boundary_trail_pts), c='tab:purple', ls='--')

    # Plot the original boundary, rotated
    plot_poly(pts, colour='tab:blue')
    # Plot the boundary to the original rotated boundary
    # boundary_pts = get_boundary_pts(pts)
    # plot_poly(boundary_pts, colour='tab:orange', ls=':')
    # boundary_trail_pts.append(np.mean(boundary_pts, axis=1))
    # ax.plot(*zip(*boundary_trail_pts), c='tab:orange', ls=':')

    # plt.savefig('frames/bbrot-{:03d}.png'.format(i+1), dpi=DPI)

    obj_pts = rotate_points(obj_pts, theta, rc)
    pts = rotate_points(pts, theta, rc)
    plt.show()
'''
