import freenect
import cv2
import numpy as np
import time
# import serial
import math
import matplotlib.mlab as mlab
from matplotlib import pyplot as plt
import sys
import copy


def start():
    global XAXIS
    global YAXIS
    global TEST_CASES
    global DIRECTION
    global GLOBAL_DEPTH_MAP
    plt.ion()
    plt.figure()
    ctx = freenect.init()
    dev = freenect.open_device(ctx, freenect.num_devices(ctx) - 1)
    freenect.set_tilt_degs(dev, 0)
    freenect.close_device(dev)
    DIRECTION = 0
    for i in range(5):
        initial_map = get_depth()
        #print("test")

    while True:
        GLOBAL_DEPTH_MAP = get_depth()	#returns the depth frame
        # back_movement(GLOBAL_DEPTH_MAP)
        # contoursright = contours_return(GLOBAL_DEPTH_MAP, -10)
        # contoursleft = contours_return(GLOBAL_DEPTH_MAP, 10)
        # door_detection(contoursright, contoursleft)
        # if DOOR_FLAG:
        #     door_movement()
        # else: regular_movement()

        #gdepth = GLOBAL_DEPTH_MAP[GLOBAL_DEPTH_MAP > 0] #ignores closest points <500mm

        #gdepth = copy.deepcopy(GLOBAL_DEPTH_MAP[20:40][0:200])
        ymin = 200
        ymax = 400
        xmin = 400
        xmax = 500
        minrange = 150

        gdepth = GLOBAL_DEPTH_MAP[ymin:ymax, xmin:xmax]

        gdepth = gdepth[gdepth > minrange]

        gdepth_mins = np.partition(gdepth,100)[:100] #sample 100 closest points
      
        
        #print(gdepth)
        #print(gdepth_mins)

        print("estimated distance:",np.mean(gdepth_mins)) #avg closest points for approx distance
        print("test")
        cv2.imshow('final', GLOBAL_DEPTH_MAP)

        cv2.waitKey(1 )
        print()
        # if cv2.waitKey(1) != -1:
        #     SERIALDATA.write('\x35')
        #     SERIALDATA.close()
        #     break

def get_depth():
    """
    * Function Name:	get_depth
    * Input:		None
    * Output:		Returns the depth information from pixel values of 0 to 255
    * Logic:		It recieves the depth information from the Kinect sensor in mm.
                    The depth range is 40cm to 800cm. The values are brought
                    down from 0 to 255. It then changes the data type
                    to 1 bytes. It then smoothens the frame and returns it.
    * Example Call:	get_depth()
    """
    global GLOBAL_DEPTH_MAP
    # depth_array = freenect.sync_get_depth(format=freenect.DEPTH_MM)[0]
    depth_array = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)[0]
    # depth_array = depth_array/30.0
    # depth_array = filter_smooth(depth_array)
    # depth_array = depth_array.astype(np.uint8)
    
    # depth_array[0:479, 630:639] = depth_array[0:479, 620:629]
    GLOBAL_DEPTH_MAP = depth_array
    return depth_array

def filter_noise(depth_array, mask, masked_array, row, col):
    """
    * Function Name:filter_noise
    * Input:		Original frame, noise mask, Original
                    frame with noise pixels being made to 255 value,
                    no. of row tiles, No. of column tiles.
    * Output:		Filters the noise from the original depth frame.
    * Logic:		The function divides rows and cols of the frame in
                    some number of pixels. It then finds the mean of the
                    tile and assigns the value to the noise pixels in that
                    tile.
    * Example Call:	filter_noise(depth_array, mask, ad, 3, 4)
    """
    row_ratio = 480/row
    column_ratio = 640/col
    temp_y = 0
    for i in range(col):
        temp_x = 0
        for j in range(row):
            area = masked_array[temp_x:temp_x+row_ratio-1, \
                   temp_y:temp_y+column_ratio-1]
            mask[temp_x:temp_x+row_ratio-1, temp_y:temp_y+column_ratio-1] \
                *= area.mean()
            depth_array[temp_x:temp_x+row_ratio-1, \
            temp_y:temp_y+column_ratio-1] += \
                mask[temp_x:temp_x+row_ratio-1, temp_y:temp_y+column_ratio-1]
            temp_x = temp_x + row_ratio
        temp_y = temp_y + column_ratio
    return depth_array

def filter_smooth(depth_array):
    """
    * Function Name:	filter_smooth
    * Input:		Original Depth frame in mm.
    * Output:		Filters the noise from the depth frame
    * Logic:		It creates a mask for the noise. It makes
                    	all the noise pixels to 255 to send to filter noise.
                    	The output from filter noise is smoothened using
                    	bilateral filter
    * Example Call:	filter_smooth(a)
    """
    ret, mask = cv2.threshold(depth_array, 10, 255, cv2.THRESH_BINARY_INV)
    mask_1 = mask/255
    masked_array = depth_array + mask
    blur = filter_noise(depth_array, mask_1, masked_array, 3, 4)
    blur = cv2.bilateralFilter(blur, 5, 50, 100)
    return blur

if __name__ == "__main__":
    start()
    # cv2.destroyAllWindows()
