from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import numpy
from time import sleep

class Camera:
    current_image = None

    def __init__(self):
        self.camera = PiCamera()
        self.camera.framerate = 32
        self.camera.resolution = (1296,976)
        self.raw_capture = PiRGBArray(self.camera, size=(1296,976))
        sleep(0.1)

    def next(self):
        return self.camera.capture_continuous(self.raw_capture, format="bgr", use_video_port=True)

    def prepare(self, frame):
        image = frame.array
        self.raw_capture.truncate(0)
        self.current_image = image
        return image

    def shutdown(self):
        self.camera.close()