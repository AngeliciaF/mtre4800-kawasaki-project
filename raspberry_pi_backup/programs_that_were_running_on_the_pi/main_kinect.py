import cv2
# from AImodel import AIModel
from Yolov5Model import Yolov5Model
# import cameraInterface
from payload_kinect import Payload
import segmentation2_kinect as segmentation
# from segmentation2_kinect import (getPayload, getPayloads)
from time import time as now
from time import sleep
import messaging_kinect
import numpy
import videoServer_kinect as flask
import threading
import multiprocessing
import math
from copy import deepcopy
from math import sqrt

# import yolov5.kinect_detect
from experimental_kinect import attempt_load
from DetectMultiBackend import DetectMultiBackend
import torch
import torch.nn as nn
# import freenect

'''
x = numpy.zeros((3,15,20))
x1 = numpy.reshape(x[0], (1,300))
x2 = numpy.reshape(x[1], (1,300))
x3 = numpy.reshape(x[2], (1,300))
y = numpy.concatenate((x1, x2, x3), 1)
print(x.shape, y.shape)
exit()
'''

# TODO: Look into moving this file (main_kinect.py) to yolov5 directory

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu')):
        # Usage:
        #   PyTorch:              weights = *.pt

        # from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        # pt = self.model_type(w)  # get backend
        # stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        # w = attempt_download(w)  # download if not local
        # fp16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16
        # fp16 &= (pt) and device.type != 'cpu'  # FP16

        # if pt:  # PyTorch
        model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
        # stride = max(int(model.stride.max()), 32)  # model stride
        # names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        # model.half() if fp16 else model.float()
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        # b, ch, h, w = im.shape  # batch, channel, height, width
        # self.jit = False
        # if self.pt or self.jit:  # PyTorch
            # y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
        y = self.model(im, augment=augment, visualize=visualize)
        return y if val else y[0]

        # if isinstance(y, np.ndarray):
        #     y = torch.tensor(y, device=self.device)
        # return (y, []) if val else y

    def warmup(self, imgsz=(1, 3, 640, 640)):
        # Warmup model by running inference once
        # if any((self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb)):  # warmup types
        if self.device.type != 'cpu':  # only warmup GPU models
            # im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            im = torch.zeros(*imgsz, dtype=torch.float, device=self.device)  # input
            # for _ in range(2 if self.jit else 1):  #
            #     self.forward(im)  # warmup

# Start the video server
# TODO: Uncomment
# threading.Thread(target=lambda: flask.main()).start()

# Get predictions from class
#model = AIModel((1296,976), 440)
# model = AIModel()
# TODO: Make a function that returns predictions
yolo_model = Yolov5Model()

weights = 'mtre4800-kawasaki-project/best.pt'
device = torch.device('cpu') # TODO: Use the select_device()
print("Main")

# Load the ML model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='mtre4800-kawasaki-project/best.pt')  # local model
# model = DetectMultiBackend(weights, device=device)

# TODO: Set the heights from the Kinect distance program here
# TODO: Add the bucket height
# FIXME Ask Tim: Which is which? ['black', 'amazon', 'clear', 'styro', 'null']?
# Order of heights are based on the training/labelling order
# From 24 in. from the ground (tallest container can be 24 in. in the future with this method)
heights = [-254, -374, -343, -275, 0]
# heights = [-254, -275, bucket, 0]

template = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
template = cv2.resize(template,(648,488))
template = template[60:480,70:610]
template = cv2.resize(template, (160,120))

scada = {'robot_tags':{'home':True}}
prediction = 3 #4

# while True:
# while rgb_image:
    # rgb_image, _ = freenect.sync_get_video()
frame = 0
for frame in range(0, 100):
    # Get the RGB image from the Kinect
    # TODO: Uncomment for live Kinect video feed
    # rgb_image, _ = freenect.sync_get_video()
    rgb_image = cv2.imread('mtre4800-kawasaki-project/three_containers4.jpg')
    rgb_image = cv2.resize(rgb_image, (640, 480))
    frame += 1

    start = now()
    # img = cam.prepare(frame)

    # # Change test frame from RGB to BGR
    # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # FIXME: Work on this
    # TODO: Convert this, too
    payloads, bounding_box = segmentation.getPayloads(rgb_image)

    print("Length of payloads:", len(payloads))
    
    # bounding_box = None
    x = 0
    y = 0
    selected = 0
    for index, payload in enumerate(payloads):
        # FIXME: 2 AsK Tim: How does payload.selected work? Is it the box the gripper is going to pick up next?
        #                   Maybe remove since payloads are already checked
        # if payload.selected:
        if True:
            selected = index
            # bounding_box, sample = segmentation.getPayload(payload, rgb_image)
            sample = segmentation.get_sample(payload, rgb_image)
            if scada['robot_tags']['home']:
                # Get 1 prediction at a time

                # FIXME: 2 Ask Tim: Does this just return the 1st prediction, even if there are multiple boxes?
                prediction = yolo_model.getPrediction(yolo_model.model, rgb_image, payload, selected)
                # prediction = model.getPrediction(sample, payload)
                payload.z = heights[payload.type]
            else:
                # Not at home, so no prediction
                payload.type = prediction
                payload.z = heights[payload.type]
            x, y = segmentation.convert_units(payload.x, payload.y, rgb_image.shape)

    if scada['robot_tags']['home']:
        segmentation.draw_payloads(rgb_image, payloads, bounding_box, yolo_model.labels)

    tag_set = Payload().tags()
    if len(payloads) > 0:
        # FIXME: 2 Ask Tim: Why are these not in a for loop?
        payloads[selected].x = x
        payloads[selected].y = y
        cv2.putText(rgb_image, "X: " + str(round(payloads[selected].x,0)) + 'mm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(rgb_image, "Y: " + str(round(payloads[selected].y,0)) + 'mm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        tag_set = deepcopy(payloads[selected].tags())

    tag_set['number_of_payloads'] = len(payloads)

    scada = messaging_kinect.client_send('vision', tag_set, True)
    #print(scada['scada_tags'])


    # Push image to video server
    img = cv2.resize(img,(648,488))
    flask.push(img)
    cv2.imshow("final", img)
    #if (cv2.waitKey(1) & 0xFF) == 27:
    #    break

    #cycle = (1/30)-(now()-start)
    #if cycle > 0.001:
    #    sleep(cycle)

    #print(round(now()-start,3)*1000, 'mS')
#cam.shutdown()

# def attempt_download(file, repo='ultralytics/yolov5'):  # from utils.downloads import *; attempt_download()
#     # Attempt file download if does not exist
#     file = Path(str(file).strip().replace("'", ''))

#     if not file.exists():
#         # URL specified
#         name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
#         if str(file).startswith(('http:/', 'https:/')):  # download
#             url = str(file).replace(':/', '://')  # Pathlib turns :// -> :/
#             file = name.split('?')[0]  # parse authentication https://url.com/file.txt?auth...
#             if Path(file).is_file():
#                 print(f'Found {url} locally at {file}')  # file already exists
#             else:
#                 safe_download(file=file, url=url, min_bytes=1E5)
#             return file

#         # GitHub assets
#         file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
#         try:
#             response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
#             assets = [x['name'] for x in response['assets']]  # release assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
#             tag = response['tag_name']  # i.e. 'v1.0'
#         except Exception:  # fallback plan
#             assets = ['yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
#                       'yolov5n6.pt', 'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
#             try:
#                 tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
#             except Exception:
#                 tag = 'v6.0'  # current release

#         if name in assets:
#             safe_download(file,
#                           url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
#                           # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
#                           min_bytes=1E5,
#                           error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')

#     return str(file)




# class DetectMultiBackend(nn.Module):
#     # YOLOv5 MultiBackend class for python inference on various backends
#     def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False):
#         # Usage:
#         #   PyTorch:              weights = *.pt

#         # from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

#         super().__init__()
#         w = str(weights[0] if isinstance(weights, list) else weights)
#         pt = self.model_type(w)  # get backend
#         stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
#         w = attempt_download(w)  # download if not local
#         # fp16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16
#         fp16 &= (pt) and device.type != 'cpu'  # FP16

#         # if pt:  # PyTorch
#         model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
#         stride = max(int(model.stride.max()), 32)  # model stride
#         names = model.module.names if hasattr(model, 'module') else model.names  # get class names
#         model.half() if fp16 else model.float()
#         self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

#         self.__dict__.update(locals())  # assign all variables to self

#     def forward(self, im, augment=False, visualize=False, val=False):
#         # YOLOv5 MultiBackend inference
#         b, ch, h, w = im.shape  # batch, channel, height, width
#         # self.jit = False
#         # if self.pt or self.jit:  # PyTorch
#             # y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
#         y = self.model(im, augment=augment, visualize=visualize)
#         return y if val else y[0]

#         # if isinstance(y, np.ndarray):
#         #     y = torch.tensor(y, device=self.device)
#         # return (y, []) if val else y

#     def warmup(self, imgsz=(1, 3, 640, 640)):
#         # Warmup model by running inference once
#         # if any((self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb)):  # warmup types
#         if self.device.type != 'cpu':  # only warmup GPU models
#             im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
#             # for _ in range(2 if self.jit else 1):  #
#             #     self.forward(im)  # warmup

#     @staticmethod
#     def model_type(p='path/to/model.pt'):
#         # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
#         from export import export_formats
#         suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
#         check_suffix(p, suffixes)  # checks
#         p = Path(p).name  # eliminate trailing separators
#         # print("p", p)
#         # pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
#         pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
#         # print("pt", pt)
#         # xml |= xml2  # *_openvino_model or *.xml
#         # tflite &= not edgetpu  # *.tflite
#         return pt