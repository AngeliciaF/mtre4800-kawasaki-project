import math
import numpy as np
import cv2
import freenect

def initialize_kinect():
    ctx = freenect.init()
    dev = freenect.open_device(ctx, freenect.num_devices(ctx) - 1)
    freenect.set_tilt_degs(dev, 0)
    freenect.close_device(dev)

def get_depth(payload, image, selected):
    xmin = 200
    xmax = 420
    ymin = 120
    ymax = 380

    minrange = 500

    samplesize = 50
    # x, y, w, h = cv2.boundingRect(contour)
    # payload.box = (x, y, w, h)
    x1, x2, y1, y2 = payload[selected].box[0], payload[selected].box[1], payload[selected].box[0] + \
                  payload[selected].box[2], payload[selected].box[1] + payload[selected].box[3]

    box_center = (int((x1+y1)/2),int((x2+y2)/2))
    cv2.rectangle(image, (x1, x2), (y1, y2), (0, 255, 0), 2)
    # print(x1,x2,y1,y2)
    xmin = x1
    ymin = x2
    ymax = y2
    xmax = y1

    depth_array = freenect.sync_get_depth
    
    gdepth = depth_array[ymin:ymax, xmin:xmax]

    gdepth = gdepth[gdepth > minrange]

    gdepth_mins = np.partition(gdepth,samplesize)[:samplesize] #sample 100 closest points

    print("estimated distance:",np.mean(gdepth_mins)) #avg closest points for approx distance