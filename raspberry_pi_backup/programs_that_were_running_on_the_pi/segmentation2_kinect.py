import math
import time
from time import time as now
import cv2
import numpy as numpy
import numpy as np
from payload_kinect import Payload
from Yolov5Model import Yolov5Model

# def main():
#     yolo_model = Yolov5Model()
#     template = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
#     template = cv2.resize(template, (640, 480))
#     payloads = getPayloads(template)
#     print("Payloads:", payloads)
#     rgb_image = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
#     rgb_image = cv2.resize(rgb_image, (640, 480))
#     print(yolo_model.getPrediction(yolo_model.model, rgb_image, payloads))

def perspective(img):
    # assumes 640X480 resolution
    input_pts = numpy.float32([[634,0],[636,482],[28,484],[9,27]])
    output_pts = numpy.float32([[0,0],[0,488],[648,488],[648,0]])
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    img = cv2.warpPerspective(img,M,(img.shape[1], img.shape[0]),flags=cv2.INTER_LINEAR)
    return img

def get_sample(payload, image, hsize=180):
    # Sample is here now to condense functions (getPayload())
    image = cv2.resize(image[300:3500,300:3500],(640, 480))

    center = (payload.x, payload.y)
    center = adjust_sample_center(center, image.shape, hsize)
    pt1 = (center[0]-hsize, center[1]-hsize)
    pt2 = (center[0]+hsize, center[1]+hsize)
    sample = image[pt1[1]:pt2[1],pt1[0]:pt2[0]]
    return sample

# Get payloads and return payloads
def getPayloads(image):
    # FIXME: Ask Tim: Are these the appoximate dimensions of the work space?
    # TODO: Once the Kinect is mounted, recalcuate/measure this work area boundary
    # image = cv2.resize(img[130:930,150:1200],(162,122))
    # image = cv2.resize(img[100:900,100:1000],(640,480))
    image = cv2.resize(image, (640,480))
    # image = cv2.resize(image[300:3500,300:3500],(640, 480))

    # cv2.imshow("Image", image)
    # cv2.waitKey(2000)
    
    print("Image shape:", image.shape)

    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur image
    blurry = cv2.GaussianBlur(gray,(9,9), 2)

    # FIXME: Ask Tim: Why blue/gray? What is the blue/gray range used for?
    # lower_blue = numpy.array([40, 30, 30])
    # lower = numpy.array([85,85,85])
    # upper_blue = numpy.array([150, 150, 190])
    # upper = numpy.array([170, 170, 190])
    lower = 100
    upper = 155
    mask = cv2.inRange(blurry, lower, upper)

    # cv2.imshow('Mask', mask)
    # cv2.waitKey(0)

    # Find Canny edges
    # edged = cv2.Canny(mask, 100, 200)

    # Invert 
    _, inv_image = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY_INV)

    # cv2.imshow('Mask/Inv', np.vstack([mask,inv_image]))
    # cv2.waitKey(2000)

    contours, _ = cv2.findContours(inv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))

    copy = image.copy()
    contours_image = cv2.drawContours(copy, contours, -1, (255,0,255), 2)

    # TODO: Refine area
    # Check area of contours to refine payload detection
    # TODO: Adjust contourArea threshold
    refined_contours = []
    for c in contours:
        contour_area = cv2.contourArea(c)
        if contour_area >= 9000 and contour_area <= 31000:
            refined_contours.append(c)

    print("Number of Refined Contours found = " + str(len(refined_contours)))

    copy1 = image.copy()
    new_contours_image = cv2.drawContours(copy1, refined_contours, -1, (255,0,0), 2)
    
    cv2.imshow('Contours_image/New_contours_image', np.hstack([contours_image,new_contours_image]))
    cv2.waitKey(2000)

    center = (0,0)
    selected = -1
    payloads = []
    new_distance_list = []
    bounding_box = []
    # index = 0
    index = 1
    # count = 0
    for contour in refined_contours:
        # count += 1
        # print("count:", count)
        # Contours -> rectangle boundaries
        bounds = cv2.minAreaRect(contour)
        # Rectangle boundaries -> Box
        box = cv2.boxPoints(bounds)
        # Box corners -> int
        box = numpy.int0(box)

        # Subtract old center x from width/2
        oc_dx = abs(center[0]-(image.shape[1]/2))
        # Subtract old center y height/2
        oc_dy = abs(center[1]-(image.shape[0]/2))
        # Calculate old distance (pixels) from contour center to center of gripper/image
        old_distance = math.sqrt(oc_dx**2+oc_dy**2)
    
        new_center = numpy.int0(bounds[0])
        # Subtract new center x from width/2
        nc_dx = abs(new_center[0]-(image.shape[1]/2))
        # Subtract new center y height/2
        nc_dy = abs(new_center[1]-(image.shape[0]/2))
        # Calculate new distance (pixels) from contour center to center of gripper/image
        new_distance = math.sqrt(nc_dx**2+nc_dy**2)


        # if check_size_low(bounds):
        new_payload = Payload()
        new_payload.bounds = bounds
        # FIXME: 2 Ask Tim: What are 6.481, 6.557, and 6.5?
        new_payload.x = int((new_center[0]*6.481)+150)
        new_payload.y = int((new_center[1]*6.557)+130)
        new_payload.r = reference_rotation(bounds)
        # FIXME: 2 Ask Tim: Is this the distance from the center of the gripper
        #                   to the center of the container?
        new_payload.distance = new_distance*6.5 ### mm
        payloads.append(new_payload)
        # new_distance_list.append(new_payload.distance)

    # Sort distances and payload at the same time based on shortest distance
    for d in new_distance_list:
        # if new_distance < old_distance:
            

        # FIXME: 3 Ask Tim: Can you explain this to me (selected)?
        if new_distance < old_distance:
            # selected = index
            # payloads[selected].selected = 1
            # if len(payloads) == 0:
                # payloads[index].selected = 1
            payloads[index-1] = new_payload
        else:
            payloads.append(new_payload)
            bounding_box.append(box)
            center = new_center
        index += 1
    
    # FIXME: 3 Ask Tim: And this?
    if len(payloads) > 0:
        payloads[selected].selected = 1

    # cv2.imshow("imagelast", image)
    # cv2.waitKey(0)
    # box is used in drawpayload
    payloads[0].selected = 1
    return payloads, bounding_box

# FIXME: Work on this (Get bounding boxes then Draw the payloads on the image)
# Add getPayload and draw payloads to the test code at the bottom
# Returns bounding boxes and image sample
def getBoundingBox(center, frame, hsize=180): #payload
    # center = (payload.x, payload.y)
    # FIXME: Ask Tim: Can you explain this?
    # center = adjust_sample_center(center, image.shape, hsize)
    pt1 = (center[0]-hsize, center[1]-hsize)
    pt2 = (center[0]+hsize, center[1]+hsize)
    sample = frame[pt1[1]:pt2[1],pt1[0]:pt2[0]]
    sample_template = frame[pt1[1]:pt2[1],pt1[0]:pt2[0]]

    [tR, tG, tB] = cv2.split(sample_template)
    [iR, iG, iB] = cv2.split(sample)

    dR = cv2.absdiff(iR, tR)
    dG = cv2.absdiff(iG, tG)
    dB = cv2.absdiff(iB, tB)
    bl = 19
    cn = 15
    tR = cv2.adaptiveThreshold(dR,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bl,cn)
    tG = cv2.adaptiveThreshold(dG,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bl,cn)
    tB = cv2.adaptiveThreshold(dB,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bl,cn)
    # TODO: experiment with merging channels
    difference = cv2.merge([tR,tG,tB])
    #cv2.imshow('fuzzy', cv2.merge([dR,dG,dB]))
    difference = cv2.cvtColor(difference,cv2.COLOR_BGR2GRAY)
    
    k_size = 15
    kernelmatrix = np.ones((k_size, k_size), np.uint8)
    # TODO: Fix black boxes dilate add more white pixel around other white pixels based on k_size
    d = cv2.dilate(difference, kernelmatrix)
    
    fuzzy = cv2.GaussianBlur(d, (9,9), 4)
    
    contours, _ = cv2.findContours(fuzzy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    targets = []
    for contour in contours:
        bounds = cv2.minAreaRect(contour)
        # if circle
            # bounds[2]
        # if check_size_hi(bounds):
        targets.append(bounds)
    
    if len(targets) > 0:
        # if len(targets) > 1:
        #     target = checkOverlap(targets)
        # else:
        #     target = targets[0]
    
        adjusted_center = (target[0][0] + pt1[0], target[0][1] + pt1[1])
        payload.x = int(adjusted_center[0])
        payload.y = int(adjusted_center[1])
        payload.r = reference_rotation(target)
        target = (adjusted_center,target[1],target[2])
        box = cv2.boxPoints(target)
        box = np.int0(box)
        return box, sample
    else:
        return None, sample

# FIXME: Ask Tim: What are these checking for?
# FIXME: Ask Tim: Why index 1 not 0?
# If the contours are too small or too big, ignore.
def check_size_hi(bounds):
# Reduce until the payload are excluded, then put back up
    width = bounds[1][0]
    height = bounds[1][1]
    area = width*height
    return ((area < 200000) and (area > 40000)) and ((width >200) and (height > 200)) and ((height < 500) and (width < 500))

def check_size_low(bounds):
    width = bounds[1][0]
    length = bounds[1][1]
    area = width*length

# FIXME: Ask Tim: How are these dimension limits measured?
# TODO: Check the limits for these with bucket dims
# Increase until the payload are excluded, tehn put back up
    side_upper_limit = 1000
    side_lower_limit = 10
    area_upper_limit = 600000
    area_lower_limit = 100
    areaOK = ( (area < area_upper_limit) and (area > area_lower_limit) )
    widthOK = ((width < side_upper_limit ) and (width > side_lower_limit) )
    lengthOK = ( (length < side_upper_limit) and (length > side_lower_limit) )
    return (areaOK and widthOK and lengthOK)

# FIXME: Ask Time: How was this measured?
# Place caps measure in mm
# take a picture
# measure in pixel
# MS Paint
# Robot at home
def convert_units(x,y,shape):
    x = -( x - (shape[1]/2) -73)
    y = ( y - (shape[0]/2) +11)
    mm_px = 1.49
    x *= mm_px
    y *= mm_px
    return x,y

# FIXME: Ask Tim: Can you please explain this?
# Limited by sensor cable
# Uses long side to pick refernce origin
# Adjust for bucket
# -bounds to make sure the axis line up
# cv cw -> +
# robot ccw -> +
def reference_rotation(bounds):
    # FIXME: 3 Ask Tim: Why is this used? So that 0 - 90 is returned?
    adjust = 0
    width = bounds[1][0]
    height = bounds[1][1]
    if width < height:
        return (90 - bounds[2])+adjust
    else: # circle
        return (-bounds[2])+adjust

def adjust_sample_center(center, shape, sample_size):
    if center[0] < sample_size:
        adjust = sample_size - center[0]
        center = (center[0]+adjust, center[1])
    elif center[0] > shape[1]-sample_size:
        adjust = sample_size - (shape[1]-center[0])
        center = (center[0]-adjust, center[1])

    if center[1] < sample_size:
        adjust = sample_size - center[1]
        center = (center[0],center[1]+adjust)

    elif center[1] > shape[0]-sample_size:
        adjust = sample_size-(shape[0]-center[1])
        center = (center[0],center[1]-adjust)

    return center

def checkOverlap(targets):
    # TODO: Test function
    area = 0
    selection = 0
    for index, payload in enumerate(targets):
        nwidth = payload[1][0]
        nlength = payload[1][1]
        narea = nwidth*nlength
        if narea > area:
            area = narea
            selection = index
    return targets[selection]

def draw_payloads(image, payloads, bounding_box, labels):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for index, payload in enumerate(payloads):
        print("draw payloads.selected:", payload.selected)
        center = (payload.x, payload.y)
        color = (0,0,255)
        font_color = (0,0,0)
        # if payload.selected:
        if True: 
            center = (payload.x, payload.y)
            color = (0,255,0)
            if bounding_box is not None:
                # cv2.drawContours(img, [bounding_box], 0, color, 2)
                # cv2.drawContours(image, [bounding_box[index]], 0, (255, 255, 0), 2)
                cv2.drawContours(image, [bounding_box[0]], 0, (255, 255, 0), 2)
            print("center:", center)
        image = cv2.circle(image,center,4,color,3)
        # cv2.rectangle(img, (center[0]+7, center[1]-100), (center[0]+240, center[1]+80), (255,255,255), -1)
        cv2.putText(image, "D: " + str(round(payload.distance,0))+'px', (center[0]+10, center[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        cv2.putText(image, "X: " + str(round(payload.x,0)) + 'px', (center[0]+10, center[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        cv2.putText(image, "Y: " + str(round(payload.y,0)) + 'px', (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        cv2.putText(image, "R: " + str(round(payload.r,0)) + 'deg', (center[0]+10, center[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        cv2.putText(image, labels[payload.type], (center[0]+10, center[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        cv2.line(image, (int(image.shape[1]/2),int(image.shape[0]/2)), center, color, 1)
    cv2.imshow("Draw Payloads", image)
    cv2.waitKey(0)

# if __name__ == "__main__":
#     main()

# ADD Payload.selected here
# def main(self):

# Test seg code
'''
heights = [-254, -374, -343, -275, 0]

scada = {'robot_tags':{'home':True}}

yolo_model = Yolov5Model()
# rgb_image, _ = freenect.sync_get_video()
rgb_image = cv2.imread('mtre4800-kawasaki-project/three_containers4.jpg')[..., ::-1]
# bgr_image = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
# rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)
# cv2.imshow("final", rgb_image)
# cv2.waitKey(0)

payloads = getPayloads(rgb_image)
print("Payloads:", payloads)

bounding_box = None
x = 0
y = 0
selected = 0
for index, payload in enumerate(payloads):
    # print("payload.selected", payload.selected)
    if payload.selected:
        selected = index
        # cv2.imshow("rgb_image", rgb_image)
        # cv2.waitKey(0)
        bounding_box, sample = getBoundingBox(payload, rgb_image)
        if scada['robot_tags']['home']:
            # prediction = model.getPrediction(sample, payload)
            predicted_labels = yolo_model.getPrediction(yolo_model.model, rgb_image, payload)
            payload.z = heights[payload.type]
        else:
            payload.type = predicted_labels[index] # Maybe add index?
            # Set heights
            payload.z = heights[payload.type]
        # Convert pixels to mm
        x, y = convert_units(payload.x, payload.y, rgb_image.shape)

# rgb_image = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
# rgb_image = cv2.resize(rgb_image, (640, 480))
print("getPrediction:", predicted_labels)

if scada['robot_tags']['home']:
    draw_payloads(rgb_image, payloads, bounding_box, yolo_model.labels)
'''


