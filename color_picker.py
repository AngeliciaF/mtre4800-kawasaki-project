# Color picker from live grayscaled videofeed for mouse click and trackbar/slider
# For Kinect Camera

import freenect
import cv2
import numpy as np

image_hsv = None    # global
pixel = (0, 0, 0)   # a default pixel
max_value = 255
low_V = 0
high_V = max_value
window_capture_name = 'RGB & HSV Video Capture'
window_depth_name = 'Depth Image'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'
trackbar_V_name = 'H:0 S:0 V'
bgr_image_resolution = (640, 480) # (1280, 720) or (416,416)
depth_image_resolution = (640, 480) # (1280, 720) or (416,416)

# mouse callback function
def pick_color_mouse2(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv_image[y,x]
        # print("y", type(y))
        # print("x", type(x))
        print("y", y)
        print("x", x)
        print("Pixel", pixel)
        print(type(pixel))
        # print("Shape: ", pixel.shape)
        #you might want to adjust the ranges(+-10, etc):
        # upper =  np.array([pixel[0] + 20, pixel[1] + 20, pixel[2] + 40])
        # lower =  np.array([pixel[0] - 20, pixel[1] - 20, pixel[2] - 40])
        upper =  np.array([pixel[0], pixel[1], pixel[2] + 5])
        lower =  np.array([pixel[0], pixel[1], pixel[2] - 5])
        # print('Pixel HSV:\t', pixel, '\tLower HSV Bound:\t', lower, '\tUpper HSV Bound:\t', upper)
        print(f"Pixel HSV:\t {pixel} \tLower HSV Bound:\t {lower} \tUpper HSV Bound:\t {upper}")

        image_mask = cv2.inRange(hsv_image,lower,upper)
        cv2.imshow("mask",image_mask)

def pick_color_mouse(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_HSV_color = depth_image[y, x]
        print(pixel_HSV_color)
        V_value = cv2.getTrackbarPos(trackbar_V_name, window_detection_name)
        # print("pos", type(cv2.getTrackbarPos(trackbar_V_name, window_detection_name)))
        # print("H", H_value)
        # print("S", S_value)
        # print("V", V_value)
        # H_value = y
        # S_value = x
        # V_value = cv2.getTrackbarPos(trackbar_V_name, window_detection_name)
        print(type(pixel_HSV_color))
        print(pixel_HSV_color)
        # Print pixel color (HSV)
        print(f'Pixel HSV: {pixel_HSV_color}\t\tH: {pixel_HSV_color[0]} S: {pixel_HSV_color[1]} V: {pixel_HSV_color[2]}\n')

        # Search for pixels within a specific range from the selected pixel value
        upper =  np.array([pixel_HSV_color[0], pixel_HSV_color[1], pixel_HSV_color[2] + 5])
        lower =  np.array([pixel_HSV_color[0], pixel_HSV_color[1], pixel_HSV_color[2] - 5])

        # Search for just the single selected pixel value
        # upper =  np.array([pixel[0], pixel[1], pixel[2]])
        # lower =  np.array([pixel[0], pixel[1], pixel[2]])
        print('Pixel HSV:\t', pixel_HSV_color, '\tLower HSV Bound:\t', lower, '\tUpper HSV Bound:\t', upper)

        # Display all the pixels with a specific color (HSV: [0, 0, ???]) based on the trackbar position
        frame_mask = cv2.inRange(hsv_image, lower, upper)
        cv2.imshow(window_detection_name, frame_mask)

# Slider callback function
def on_V_thresh_trackbar(val):
   global low_V
   global high_V
   high_V = val
   high_V = max(high_V, low_V+1)
   cv2.setTrackbarPos(trackbar_V_name, window_detection_name, high_V)

def main():
    global hsv_image, depth_image, pixel, value, window_detection_name, y,x # so we can use it in mouse callback
    y = 0
    x = 0
    value = 0
    # Select the camera to capture video from
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Create the window for the hsv image
    # cv2.namedWindow('hsv')

    # Create the window for the video capture
    cv2.namedWindow(window_capture_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_capture_name, bgr_image_resolution)

    # Create the window for the depth image
    cv2.namedWindow(window_depth_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_depth_name, depth_image_resolution)

    # Create the window for the trackbar/slider detection
    cv2.namedWindow(window_detection_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_detection_name, bgr_image_resolution)

    # Create the trackbar/slider
    cv2.createTrackbar(trackbar_V_name, window_detection_name , low_V, max_value, on_V_thresh_trackbar)

    cv2.setMouseCallback(window_depth_name, pick_color_mouse)
    while True:
        # Get the frames from the webcam
        # _, frame = cap.read()

        # Get the RGB image from the Kinect
        rgb_image, _ = freenect.sync_get_video()

        # Make sure an RGB frame was read
        if rgb_image is None:
            print ("No RGB frame was read.")

        # TODO: Check to make sure the image can be RGB instead of BGR
        # Convert image from the RGB to BGR to HSV and resizing the image from the Kinect
        rgb_image = rgb_image.astype(np.uint8)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV_FULL)
        hsv_image = cv2.resize(hsv_image, bgr_image_resolution, interpolation= cv2.INTER_AREA)        
        # bgr_image = cv2.resize(bgr_image, bgr_image_resolution, interpolation= cv2.INTER_AREA)        

        # Convert from BGR to HSV_FULL (H: 0-255, S: 0-255, V: 0-255)
        # frame_hsv_full = cv2.cvtColor(hsv_image, cv2.COLOR_BGR2HSV_FULL)
        # cv2.imshow("HSV_FULL", frame_hsv_full)

        # Get the depth image from the Kinect
        depth_image, _ = freenect.sync_get_depth()

        # Make sure a depth frame was read
        if depth_image is None:
            print ("No depth frame was read.")
            return

        # TODO: 
        # Resize the depth image from the Kinect
        # Changes the max values from 255 to 
        # depth_image = depth_image.astype(np.uint16)
        depth_image = depth_image.astype(np.uint8)
        depth_image = cv2.resize(depth_image, depth_image_resolution, interpolation= cv2.INTER_AREA)

        # Make the 2D depth_image 3D (640, 480) --> (640, 480, 3)
        # depth_image = depth_image[:, :, np.newaxis]
        depth_image = np.dstack([depth_image]*3)
        # print(depth_image)
        # print(depth_image.shape)

        # print("rgb image shape: ", rgb_image.shape)
        # print("hsv image shape: ", hsv_image.shape)
        # print("Depth image shape: ", depth_image.shape)
        # cv2.imshow("Depth image", depth_image)
        # Display the BGR videofeed
        cv2.imshow(window_depth_name, depth_image)

        cv2.imshow(window_capture_name, np.hstack([rgb_image, hsv_image, depth_image]))
        # cv2.imshow(window_capture_name, np.hstack([rgb_image, hsv_image]))
        
        # # if event == cv2.EVENT_LBUTTONDOWN:

        # H_value, S_value, V_value = y, x, cv2.getTrackbarPos(trackbar_V_name, window_detection_name)
        # # S_value = x
        # # V_value = cv2.getTrackbarPos(trackbar_V_name, window_detection_name)
        # pixel = hsv_image[H_value, S_value, V_value]

        # print(pixel)
        # # Print pixel color (HSV)
        # print(f'Pixel HSV: {pixel}\t\tH: {pixel[0]} S: {pixel[1]} V: {pixel[2]}\n')

        # # Search for pixels within a specific range from the selected pixel value
        # upper =  np.array([pixel[0], pixel[1], pixel[2] + 5])
        # lower =  np.array([pixel[0], pixel[1], pixel[2] - 5])

        # # Search for just the single selected pixel value
        # # upper =  np.array([pixel[0], pixel[1], pixel[2]])
        # # lower =  np.array([pixel[0], pixel[1], pixel[2]])
        # print('Pixel HSV:\t', pixel, '\tLower HSV Bound:\t', lower, '\tUpper HSV Bound:\t', upper)

        # # Display all the pixels with a specific color (HSV: [0, 0, ???]) based on the trackbar position
        # frame_mask = cv2.inRange(hsv_image, lower, upper)
        # cv2.imshow(window_detection_name, frame_mask)

        # Convert from BGR to HSV (H: 0-179, S: 0-255, V: 0-255)
        # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV", frame_hsv)

        # Display live videofeed and end live videofeed with 'q' and esc
        if cv2.waitKey(30) == ord('q') or cv2.waitKey(30) == 27:
            break
    
        # Step through live videofeed with "any key" (Using the Spacebar is the easiest.)
        # cv2.waitKey(0)

# Stop receiving frames from the Kinect camera
def close_camera():
    # cap.release()
    cv2.destroyAllWindows()
    print("Closed Video Capture")
if __name__=='__main__':
    main()
    close_camera()

# Color picker from grayscaled image for mouse click and trackbar/slider
# For Windows webcam
'''
import sys
import cv2
import numpy as np

image_hsv = None    # global
pixel = (0, 0, 0)   # a default pixel
max_value = 255
low_V = 0
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'
trackbar_V_name = 'H:0 S:0 V'
image_resolution = (1280, 720)

# mouse callback function
def pick_color_mouse(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 20, pixel[1] + 20, pixel[2] + 40])
        lower =  np.array([pixel[0] - 20, pixel[1] - 20, pixel[2] - 40])
        print('Pixel HSV:\t', pixel, '\tLower HSV Bound:\t', lower, '\tUpper HSV Bound:\t', upper)

        image_mask = cv2.inRange(image_hsv, lower, upper)
        cv2.imshow("mask", image_mask)

# slider callback function
def on_low_V_thresh_trackbar(val):
   global low_V
   global high_V
   low_V = val
   low_V = min(high_V-1, low_V)
   cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
   global low_V
   global high_V
   high_V = val
   high_V = max(high_V, low_V+1)
   cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)
def on_V_thresh_trackbar(val):
   global low_V
   global high_V
   high_V = val
   high_V = max(high_V, low_V+1)
   cv2.setTrackbarPos(trackbar_V_name, window_detection_name, high_V)
def pick_color_trackbar(event, x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # pixel = image_hsv[y,x]
        val = cv2.getTrackbarPos(trackbar_V_name, window_detection_name)
        pixel = [0, 0, val]

        # Print pixel color (HSV)
        print(f'Pixel HSV: {pixel}\t\tH: {pixel[0]} S: {pixel[1]} V: {pixel[2]}\n')

        # Search for pixels within a specific range from the selected pixel value
        upper =  np.array([pixel[0], pixel[1], pixel[2] + 30])
        lower =  np.array([pixel[0], pixel[1], pixel[2] - 30])

        # Search for just the single selected pixel value
        # upper =  np.array([pixel[0], pixel[1], pixel[2]])
        # lower =  np.array([pixel[0], pixel[1], pixel[2]])
        # print('Pixel HSV:\t', pixel, '\tLower HSV Bound:\t', lower, '\tUpper HSV Bound:\t', upper)

        # Display all the pixels with a specific color (HSV: [0, 0, ???]) based on the trackbar position
        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow(window_detection_name,image_mask)


def main():
    global image_hsv, pixel, val, window_detection_name # so we can use it in mouse callback

    # image_src_original = cv2.imread('rainbow1.jpg')
    # image_src_original = cv2.imread('orange_container1.jpg')
    # image_src_original = cv2.imread('white_container1.jpg')
    # image_src_original = cv2.imread('black_container1.jpg')
    # image_src_original = cv2.imread('three_containers1.jpg')
    image_src_original = cv2.imread('depth_image1.png')
    image_src = cv2.resize(image_src_original, image_resolution)

    # Make sure an image was read
    if image_src is None:
        print ("No image was read.")
        return
    # cv2.imshow("bgr",image_src)

    # Create the window for the hsv, trackbar/slider detection, and video capture
    cv2.namedWindow('hsv')
    cv2.namedWindow(window_detection_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_detection_name, image_resolution)
    # cv2.namedWindow(window_capture_name)

    cv2.createTrackbar(trackbar_V_name, window_detection_name , low_V, max_value, on_V_thresh_trackbar)


    # Clicking left mouse button on the hsv window calls pick_color function 
    # cv2.setMouseCallback('hsv', pick_color_mouse)
    cv2.setMouseCallback('hsv', pick_color_trackbar)

    # Clicking left mouse button on the window_detection_name window calls pick_color function 
    cv2.setMouseCallback(window_detection_name, pick_color_trackbar)

    # pick_color_mouse: Click the hsv img to look at the pixel values
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv",image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
'''

# Color picker from live grayscaled video feed for mouse click and trackbar/slider
# For Windows webcam
'''
import sys
import cv2
from cv2 import CAP_DSHOW
import numpy as np

image_hsv = None    # global
pixel = (0, 0, 0)   # a default pixel
max_value = 255
low_V = 0
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_V_name = 'Low V'
high_V_name = 'High V'
trackbar_V_name = 'H:0 S:0 V'
image_resolution = (640, 480) # (1280, 720)

# mouse callback function
def pick_color_mouse(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 20, pixel[1] + 20, pixel[2] + 40])
        lower =  np.array([pixel[0] - 20, pixel[1] - 20, pixel[2] - 40])
        print('Pixel HSV:\t', pixel, '\tLower HSV Bound:\t', lower, '\tUpper HSV Bound:\t', upper)

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("mask",image_mask)

# Slider callback function
def on_V_thresh_trackbar(val):
   global low_V
   global high_V
   high_V = val
   high_V = max(high_V, low_V+1)
   cv2.setTrackbarPos(trackbar_V_name, window_detection_name, high_V)

def main():
    global frame_hsv_full, pixel, value, window_detection_name, cap # so we can use it in mouse callback

    # Select the camera to capture video from
    cap = cv2.VideoCapture(0, CAP_DSHOW)

    # Create the window for the hsv image
    # cv2.namedWindow('hsv')

    # Create the window for the trackbar/slider detection
    cv2.namedWindow(window_detection_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_detection_name, image_resolution)

    # Create the window for the video capture
    cv2.namedWindow(window_capture_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_capture_name, image_resolution)

    # Create the trackbar/slider
    cv2.createTrackbar(trackbar_V_name, window_detection_name , low_V, max_value, on_V_thresh_trackbar)

    while True:
        _, frame = cap.read()
        # Make sure a frame was read
        if frame is None:
            print ("No frame was read.")
            return

        # Display the BGR videofeed
        # cv2.imshow(window_capture_name, frame)

        # Convert from BGR to HSV_FULL (H: 0-255, S: 0-255, V: 0-255)
        frame_hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
        # cv2.imshow("HSV_FULL", frame_hsv_full)
        cv2.imshow(window_capture_name, np.hstack([frame, frame_hsv_full]))
        # cv2.imshow(window_capture_name, np.vstack([np.hstack([frame, frame_hsv_full]), frame_mask]))
        
        # pixel = image_hsv[y,x]
        value = cv2.getTrackbarPos(trackbar_V_name, window_detection_name)
        pixel = [0, 0, value]

        # Print pixel color (HSV)
        print(f'Pixel HSV: {pixel}\t\tH: {pixel[0]} S: {pixel[1]} V: {pixel[2]}\n')

        # Search for pixels within a specific range from the selected pixel value
        upper =  np.array([pixel[0] + 20, pixel[1] + 20, pixel[2] + 30])
        lower =  np.array([pixel[0] - 20, pixel[1] - 20, pixel[2] - 30])

        # Search for just the single selected pixel value
        # upper =  np.array([pixel[0], pixel[1], pixel[2]])
        # lower =  np.array([pixel[0], pixel[1], pixel[2]])
        # print('Pixel HSV:\t', pixel, '\tLower HSV Bound:\t', lower, '\tUpper HSV Bound:\t', upper)

        # Display all the pixels with a specific color (HSV: [0, 0, ???]) based on the trackbar position
        frame_mask = cv2.inRange(frame_hsv_full, lower, upper)
        cv2.imshow(window_detection_name, frame_mask)


        # Convert from BGR to HSV (H: 0-179, S: 0-255, V: 0-255)
        # frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV", frame_hsv)

        # Display live videofeed and end live videofeed with 'q' and esc
        if cv2.waitKey(30) == ord('q') or cv2.waitKey(30) == 27:
            break
    
        # Step through live videofeed with "any key" (Using the Spacebar is the easiest.)
        # cv2.waitKey(0)

def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
    print("Closed Video Capture")
if __name__=='__main__':
    main()
    close_camera(cap)
'''
