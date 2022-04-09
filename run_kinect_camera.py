# Run the Kinect camera
from cv2 import WINDOW_NORMAL
import freenect
import cv2
from matplotlib.colors import rgb2hex
import numpy as np

def main():
    print("Starting the Kinect Camera")
    rgb_window_name = "RGB Video Feed"
    depth_window_name = "Depth Video Feed"
    cv2.namedWindow(rgb_window_name, WINDOW_NORMAL)
    cv2.namedWindow(depth_window_name, WINDOW_NORMAL)

    while True:
        # Get the RGB image from the Kinect
        rgb_image, _ = freenect.sync_get_video()
        rgb_image = rgb_image.astype(np.uint8)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        print(bgr_image.shape)
        print(bgr_image.shape[0])
        print(bgr_image.shape[1])
        # print(rgb_image)
        # Get the depth image from the Kinect
        depth_image, _ = freenect.sync_get_depth()
        depth_image = depth_image.astype(np.uint8)

        # Display RGB and Depth Image
        cv2.imshow(rgb_window_name, bgr_image)
        cv2.imshow(depth_window_name, depth_image)

        # End live video feed with 'q' or Esc
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
            break
        #cv2.waitKey(0)

def close_camera():
    cv2.destroyAllWindows()
    print("Closed Video Capture")

if __name__ == "__main__":
    main()
    close_camera()