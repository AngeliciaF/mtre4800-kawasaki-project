# Test/Run the Kinect camera
import freenect
import cv2
from matplotlib.colors import rgb2hex
import numpy as np

def main():
    print("Starting the Kinect Camera")
    while True:
        # Get the RGB image from the Kinect
        rgb_image, _ = freenect.sync_get_video()
        rgb_image = rgb_image.astype(np.uint8)
        print(rgb_image.shape)
        print(rgb_image.shape[0])
        print(rgb_image.shape[1])
        # print(rgb_image)
        # Get the depth image from the Kinect
        depth_image, _ = freenect.sync_get_depth()
        depth_image = depth_image.astype(np.uint8)

        # Display RGB and Depth Image
        cv2.imshow("RGB Image", rgb_image)
        cv2.imshow("Depth Image", depth_image)

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