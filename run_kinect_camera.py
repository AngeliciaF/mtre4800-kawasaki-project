# Run the Kinect camera
import freenect
import cv2
from matplotlib.colors import rgb2hex
import numpy as np
# import OpenNI
# import OpenNI2
# from primesense import openni2
from openni import openni2

# OpenNI.toggleImageAutoExposure()
# OpenNI2.toggleImageAutoExposure()
# exposure = -1
# openni::CameraSettings* pCamSettings = pStreamImage.getCameraSettings()
# if (pCamSettings):
#     exposure = pCamSettings.getExposure()


def main():
    print("Starting the Kinect Camera")
    rgb_window_name = "RGB Video Feed"
    depth_window_name = "Depth Video Feed"
    cv2.namedWindow(rgb_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(depth_window_name, cv2.WINDOW_NORMAL)

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
        # cv2.imshow("image", bgr_image)
        # path = str(f'/home/user/code/mtre4800-kawasaki-project/pixel_to_mm_ratio1.jpg')
        # cv2.imwrite(path, bgr_image)

        # End live video feed with 'q' or Esc
        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
            break
        #cv2.waitKey(0)

def test():
    # OpenNI.
    # openni2.initialize("./Redist")     # can also accept the path of the OpenNI redistribution
    # openni2.initialize("/home/user/code/OpenNI/Platform/Linux/CreateRedist/Redist_OpenNI.py")     # can also accept the path of the OpenNI redistribution
    # openni2.initialize("/home/user/code/OpenNI/Platform/Linux/Bin/x64-Release/")     # can also accept the path of the OpenNI redistribution
    openni2.initialize("/home/user/code/OpenNI-Linux-x64-2.2.0.33/OpenNI-Linux-x64-2.2/Redist/")     # can also accept the path of the OpenNI redistribution
    # openni2.initialize("/home/user/code/OpenNI2/Packaging/Linux/Redist")     # can also accept the path of the OpenNI redistribution
    # openni2.initialize()     # can also accept the path of the OpenNI redistribution

    dev = openni2.Device.open_any()
    # a = openni2.CameraSettings().auto_exposure
    # b = openni2.CameraSettings().auto_white_balance
    # c = openni2.CameraSettings().exposure
    # d = openni2.CameraSettings().gain
    e = openni2.CameraSettings.set_auto_exposure()
    f = openni2.CameraSettings().exposure
    g = openni2.CameraSettings().gain
    h = openni2.CameraSettings().set_auto_white_balance()
    i = openni2.CameraSettings().set_gain()


    depth_stream = dev.create_depth_stream()
    depth_stream.start()

    while(True):

        frame = depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()

        img = np.frombuffer(frame_data, dtype=np.uint16)
        img.shape = (1, 480, 640)
        img = np.concatenate((img, img, img), axis=0)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)

        cv2.imshow("image", img)
        cv2.waitKey(0)


    depth_stream.stop()
    openni2.unload()

def close_camera():
    cv2.destroyAllWindows()
    print("Closed Video Capture")

if __name__ == "__main__":
    main()
    # test()
    close_camera()

