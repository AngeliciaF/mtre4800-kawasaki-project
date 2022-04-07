import math
# import time
# from time import time as now
import cv2
import numpy as numpy
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

# Get payloads and return payloads
def getPayload(img):
    # FIXME: Ask Tim: Are these the appoximate dimensions of the work space?
    # TODO: Once the Kinect is mounted, recalcuate/measure this work area boundary
    # Find farthest boundary and measure in pixels
    # measure with tape measureer
    # pixel to mm conversion (pixel to mm)
    # Extra testing for the black boxes
    # continuously adjust pixel to mm
    # Initial measruement from home then above the box (move just x and y)
    # image = cv2.resize(img[130:930,150:1200],(162,122))
    # image = cv2.resize(img[100:900,100:1000],(640,480))
    image = cv2.resize(img,(640,480))
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    # FIXME: Ask Tim: Why blue? What is the blue range used for?
    # remove floor and not payload
    # prefer to have false negative than false positives
    lower_blue = numpy.array([40, 30, 30])
    upper_blue = numpy.array([170, 170, 190])
    image = cv2.inRange(image, lower_blue, upper_blue)
    # cv2.imshow("image0", image)
    # cv2.waitKey(0)

    image = cv2.GaussianBlur(image,(9,9), 4)
    # cv2.imshow("image1", image)
    # cv2.waitKey(0)
    _, image = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    cv2.imshow("image2", image)
    cv2.waitKey(0)

    # TODO: Put the houghcircles here
    center = (0,0)
    selected = -1
    payloads = []
    index = 0
    count = 0
    for contour in contours:
        count += 1
        print("count:", count)
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
        # Calculate old distance
        old_distance = math.sqrt(oc_dx**2+oc_dy**2)
    
        new_center = numpy.int0(bounds[0])
        # Subtract new center x from width/2
        nc_dx = abs(new_center[0]-(image.shape[1]/2))
        # Subtract new center y height/2
        nc_dy = abs(new_center[1]-(image.shape[0]/2))
        # Calculate new distance
        new_distance = math.sqrt(nc_dx**2+nc_dy**2)

        if check_size_low(bounds):
            new_payload = Payload()
            new_payload.bounds = bounds
            new_payload.x = int((new_center[0]*6.481)+150)
            new_payload.y = int((new_center[1]*6.557)+130)
            new_payload.r = reference_rotation(bounds)
            new_payload.distance = new_distance*6.5
            payloads.append(new_payload)

            if new_distance < old_distance:
                selected = index
                center = new_center
            index += 1
    
    if len(payloads) > 0:
        payloads[selected].selected = 1
    return payloads

# FIXME: Work on this (Get bounding boxes then Draw the payloads on the image)
# Add getPayload and draw payloads to the test code at the bottom
# Returns bounding boxes and image sample
def getBoundingBox(payload, image, hsize=180):
    center = (payload.x, payload.y)
    # FIXME: Ask Tim: Can you explain this?
    # Take slices of the box
    center = adjust_sample_center(center, image.shape, hsize)
    pt1 = (center[0]-hsize, center[1]-hsize)
    pt2 = (center[0]+hsize, center[1]+hsize)
    sample = image[pt1[1]:pt2[1],pt1[0]:pt2[0]]
    sample_template = image[pt1[1]:pt2[1],pt1[0]:pt2[0]]

    [tR, tG, tB] = cv2.split(sample_template)
    [iR, iG, iB] = cv2.split(sample)

    dR = cv2.absdiff(iR, tR)
    dG = cv2.absdiff(iG, tG)
    dB = cv2.absdiff(iB, tB)
    bl = 19
    cn = 15
    tR = cv2.adaptiveThreshold(dR,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bl,cn)
    tG = cv2.adaptiveThreshold(dG,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bl,cn)
    tB = cv2.adaptiveThreshold(dB,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bl,cn)
    difference = cv2.merge([tR,tG,tB])
    cv2.imshow('fuzzy', cv2.merge([dR,dG,dB]))
    cv2.waitKey(0)
    difference = cv2.cvtColor(difference,cv2.COLOR_BGR2GRAY)
    
    k_size = 15
    kernelmatrix = numpy.ones((k_size, k_size), numpy.uint8)
    d = cv2.dilate(difference, kernelmatrix)
    
    fuzzy = cv2.GaussianBlur(d, (9,9), 4)
    
    contours, _ = cv2.findContours(fuzzy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    targets = []
    for contour in contours:
        bounds = cv2.minAreaRect(contour)
        if check_size_hi(bounds):
            targets.append(bounds)
    
    if len(targets) > 0:
        if len(targets) > 1:
            target = checkOverlap(targets)
        else:
            target = targets[0]
        
        adjusted_center = (target[0][0] + pt1[0], target[0][1] + pt1[1])
        payload.x = int(adjusted_center[0])
        payload.y = int(adjusted_center[1])
        payload.r = reference_rotation(target)
        target = (adjusted_center,target[1],target[2])
        box = cv2.boxPoints(target)
        box = numpy.int0(box)
        return box, sample
    else:
        return None, sample

# FIXME: Ask Tim: What are these checking for?
# FIXME: Ask Tim: Why index 1 not 0?
def check_size_hi(bounds):
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
    side_upper_limit = 1000
    side_lower_limit = 10
    area_upper_limit = 600000
    area_lower_limit = 100
    areaOK = ( (area < area_upper_limit) and (area > area_lower_limit) )
    widthOK = ((width < side_upper_limit ) and (width > side_lower_limit) )
    lengthOK = ( (length < side_upper_limit) and (length > side_lower_limit) )
    return (areaOK and widthOK and lengthOK)

# FIXME: Ask Time: How was this measured? Is it based on (640, 480) or (224, 192)
def convert_units(x,y,shape):
    x = -( x - (shape[1]/2) -73 )
    y = ( y - (shape[0]/2) +11 )
    mm_px = 1.49
    x *= mm_px
    y *= mm_px
    return x,y

# FIXME: Ask Tim: Can you please explain this?
def reference_rotation(bounds):
    adjust = 0
    width = bounds[1][0]
    height = bounds[1][1]
    if width < height:
        return (90 - bounds[2])+adjust
    else:
        return (-bounds[2])+adjust

# Don't touch!
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

def draw_payloads(img, payloads, bounding_box, labels):
    for index, payload in enumerate(payloads):
        print("draw payloads.selected:", payload.selected)
        center = (payload.x, payload.y)
        # Red line
        color = (0,0,255)
        # font_color = (0,0,0)
        if payload.selected:
            center = (payload.x, payload.y) 
            # Green Line
            color = (0,255,0)
            if bounding_box is not None:
                cv2.drawContours(img, [bounding_box], 0, color, 2)
            print("center:", center)
            
        cv2.circle(img,center,4,color,3)
        #cv2.rectangle(img, (center[0]+7, center[1]-100), (center[0]+240, center[1]+80), (255,255,255), -1)
        #cv2.putText(img, "D: " + str(round(payload.distance,0))+'px', (center[0]+10, center[1]-60), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        #cv2.putText(img, "X: " + str(round(payload.x,0)) + 'px', (center[0]+10, center[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        #cv2.putText(img, "Y: " + str(round(payload.y,0)) + 'px', (center[0]+10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        #cv2.putText(img, "R: " + str(round(payload.r,0)) + 'deg', (center[0]+10, center[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        #cv2.putText(img, labels[payload.type], (center[0]+10, center[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2)
        cv2.line(img, (int(img.shape[1]/2),int(img.shape[0]/2)), center, color, 1)
        cv2.imshow("final", img)
        cv2.waitKey(0)

# Draw bounding boxes
# def drawBoundingBox(color, predicted_labels, cord_thres):
def drawBoundingBox(color, frame, predicted_labels, cord_thres):
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(len(predicted_labels)):
        row = cord_thres[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, predicted_labels[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)


# if __name__ == "__main__":
#     main()

# ADD Payload.selected here
# def main(self):

heights = [-254, -374, -343, -275, 0]

scada = {'robot_tags':{'home':True}}

yolo_model = Yolov5Model()
# rgb_image, _ = freenect.sync_get_video()
rgb_image = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)
payloads = getPayload(rgb_image)
print("Payloads:", payloads)

bounding_box = None
x = 0
y = 0
selected = 0
for index, payload in enumerate(payloads):
    print("payload.selected", payload.selected)
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

# TODO: Figure out why the black box is not being detected in getprediction
# rgb_image = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
# rgb_image = cv2.resize(rgb_image, (640, 480))
print("getPrediction:", predicted_labels)

if scada['robot_tags']['home']:
    draw_payloads(rgb_image, payloads, bounding_box, yolo_model.labels)



