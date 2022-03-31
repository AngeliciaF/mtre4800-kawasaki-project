# Keep this file (n_run_kinect.py) in the yolov5 directory

import cv2
import numpy as np
# import cameraInterface
from time import time as now
from math import sqrt
# from pycoral.adapters import common
# from pycoral.adapters import classify
# from pycoral.utils.edgetpu import make_interpreter
# from picamera.array import PiRGBArray
# from picamera import PiCamera 
from PIL import Image
import pathlib
import time
import cv2
import os
import logging
import torch
import torch.nn as nn
import torchvision
import freenect
from utils.downloads import attempt_download
from utils.torch_utils import select_device
from models.common import DetectMultiBackend


VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode


center = (648, 488)

# TODO: Finish making a custom DetectMultiBackend
weights = 'mtre4800-kawasaki-project/best.pt'
# device = torch.device('cpu')
# Load the ML model
device = ''
device = select_device(device)
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='mtre4800-kawasaki-project/best.pt')  # local model
model = DetectMultiBackend(weights, device=device)


# script_dir = pathlib.Path(__file__).parent.absolute()
# model_file = os.path.join(script_dir, 'diff_models/5_1000_04-45-13-model/mnist.tflite')
# interpreter = make_interpreter(model_file)
# interpreter.allocate_tensors()
# size = common.input_size(interpreter)
# labels = ['black', 'card', 'clear', 'styro']

# TODO: Figure out what size is
size = 5

index = 232
print('starting system')
# cam = cameraInterface.Camera()

frame = 0
for frame in range(0, 100):
    # Get the RGB image from the Kinect
    # rgb_image, _ = freenect.sync_get_video()
    rgb_image = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
    rgb_image = cv2.resize(rgb_image, (416,416))

    # Change test frame from RGB to BGR
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    frame += 1

# TODO: Look into making a custom LoadStreams like in kinect_detect.py
    # # Add images from the Kinect to imgs list
    # if n % read == 0:
    #     # success, im = cap.retrieve()
    #     success, im = True, bgr_image
    #     if success:
    #         self.imgs[i] = im

    im = torch.from_numpy(bgr_image).to(device)
    # bgr_image = bgr_image.half() if model.fp16 else bgr_image.float()  # uint8 to fp16/32
    im = im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=False, visualize=False)

    # image = cam.prepare(frame)
    img = cv2.resize(bgr_image, (160, 120), cv2.INTER_AREA)
    
    template = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
    template = cv2.resize(template, (160,120), cv2.INTER_AREA)
    [tR, tG, tB] = cv2.split(template)
    [iR, iG, iB] = cv2.split(img)

    dR = cv2.absdiff(iR, tR)
    dG = cv2.absdiff(iG, tG)
    dB = cv2.absdiff(iB, tB)
    bl = 19
    cn = 15
    tR = cv2.adaptiveThreshold(dR,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bl,cn)
    tG = cv2.adaptiveThreshold(dG,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bl,cn)
    tB = cv2.adaptiveThreshold(dB,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,bl,cn)
    difference = cv2.merge([tR,tG,tB])
    difference = cv2.cvtColor(difference,cv2.COLOR_BGR2GRAY)
    
    k_size = 5
    kernelmatrix = np.ones((k_size, k_size), np.uint8)
    d = cv2.dilate(difference, kernelmatrix)
    fuzzy = cv2.GaussianBlur(d, (5,5), 2)
    contours, hierarchy = cv2.findContours(fuzzy, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    larea = 1
    for region in contours:
        bounds = cv2.minAreaRect(region)
        box = cv2.boxPoints(bounds)
        box = np.int0(box)

        width = bounds[1][0]
        height = bounds[1][1]
        #print(width, height)
        area = width*height

        if height > 20 and height < 100 and width > 20 and width < 100 and area > 5 and area < 10000 and area>larea:
            larea = area
            #image = cv2.drawContours(image,[box],0,(0,0,255),2)
            #image = cv2.rectangle(image, (0,0), (160,120), (255,255,255), 2)
            dx = abs(center[0] - bounds[0][0]*8.1)
            dy = abs(center[1] - bounds[0][1]*8.1)
            #if sqrt((dx**2+dy**2)) < 100:
            center = (int(bounds[0][0]*8.1),int(bounds[0][1]*8.1))
    box_size = 220
    if center[0] < box_size:
        adjust = box_size - center[0] - 1
        center = (center[0]+adjust,center[1])
    elif center[0] > bgr_image.shape[1]-box_size:
        adjust = center[0]-box_size - 1
        center = (center[0]-adjust,center[1])

    if center[1] < box_size:
        adjust = box_size - center[1] - 1
        center = (center[0],center[1]+adjust)
    elif center[1] > bgr_image.shape[0]-box_size:
        adjust = center[1]-box_size - 1
        center = (center[0],center[1]-adjust)
    
    point1 = (abs(int(center[0]-box_size)),abs(int(center[1]-box_size)))
    point2 = (abs(int(center[0]+box_size)),abs(int(center[1]+box_size)))
    bgr_image = bgr_image[point1[1]:point2[1],point1[0]:point2[0]]
    #cv2.rectangle(image, point1, point2, (255,255,255),3)
    bgr_image = cv2.resize(bgr_image,(640,480))

    img2 = Image.fromarray(bgr_image).convert('L').resize(size, Image.ANTIALIAS)

    # Run an inference
    # common.set_input(interpreter, img2)
    # interpreter.invoke()
    # classes = classify.get_classes(interpreter, top_k=1)
    # prediction = classes[0].score
    
    hello = ""

    container_tensor = pred[0]         

    # Get the container prediction if a prediction has been made
    if not container_tensor.numel() == 0:
        # Get container predicition
        for items in range(0, len(pred[0])):
            # print("len()", len(pred))
            # print("items",items)
            container_prediction = int(pred[0][items-1][5])

            if container_prediction == 0:
                x = 1
                print("black box")
            elif container_prediction == 1:
                x = 3
                print("orange bucket")
            elif container_prediction == 2:
                x = 2
                print("white box")
            else:
                print("Error with the container prediction")
    else:
        x = 0
        print("There are no containers.")

    bgr_image = cv2.resize(bgr_image, (640,480))
    cv2.putText(bgr_image, str(x) + " : " + hello, (40,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255),1)
    cv2.imshow("frame", bgr_image)
    if (cv2.waitKey(1) & 0xFF) == 27:
        break

# cam.shutdown()





def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            LOGGER.warning(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def set_logging(name=None, verbose=VERBOSE):
    # Sets level and returns logger
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    logging.basicConfig(format="%(message)s", level=logging.INFO if (verbose and rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)
LOGGER = set_logging('yolov5')  # define globally (used in train.py, val.py, detect.py, etc.)

# def print_args(name, opt):
#     # Print argparser arguments
#     LOGGER.info(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights='yolov5s.pt', device=torch.device('cpu'), dnn=False, data=None, fp16=False):
        # Usage:
        #   PyTorch:              weights = *.pt

        # from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        # pt = self.model_type(w)  # get backend
        pt = True  # get backend
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        w = attempt_download(w)  # download if not local
        # fp16 &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16
        fp16 &= (pt) and device.type != 'cpu'  # FP16

        # if pt:  # PyTorch
        model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
        stride = max(int(model.stride.max()), 32)  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        model.half() if fp16 else model.float()
        self.model = model  # explicitly assign for to(), cpu(), cuda(), half()

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False, val=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
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
            im = torch.zeros(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            # for _ in range(2 if self.jit else 1):  #
            #     self.forward(im)  # warmup

    # @staticmethod
    # def model_type(p='path/to/model.pt'):
    #     # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
    #     from export import export_formats
    #     suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
    #     check_suffix(p, suffixes)  # checks
    #     p = Path(p).name  # eliminate trailing separators
    #     # print("p", p)
    #     # pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
    #     pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
    #     # print("pt", pt)
    #     # xml |= xml2  # *_openvino_model or *.xml
    #     # tflite &= not edgetpu  # *.tflite
    #     return pt

def attempt_load(weights, map_location=None, inplace=True, fuse=True):
    # from models.yolo import Detect, Model
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location=map_location)  # load
        ckpt = (ckpt.get('ema') or ckpt['model']).float()  # FP32 model
        model.append(ckpt.fuse().eval() if fuse else ckpt.eval())  # fused or un-fused model in eval mode

    # Compatibility updates
    for m in model.modules():
        t = type(m)
        # if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU):
            m.inplace = inplace  # torch 1.7.0 compatibility
            # if t is Detect:
            #     if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
            #         delattr(m, 'anchor_grid')
            #         setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        # elif t is Conv:
        #     m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble
        
# def select_device(device='', batch_size=0, newline=True):
#     # device = 'cpu' or '0' or '0,1,2,3'
#     s = f'YOLOv5 🚀 {git_describe() or file_update_date()} torch {torch.__version__} '  # string
#     device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
#     cpu = device == 'cpu'
#     if cpu:
#         os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
#     elif device:  # non-cpu device requested
#         os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
#         assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
#             f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

#     cuda = not cpu and torch.cuda.is_available()
#     if cuda:
#         devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
#         n = len(devices)  # device count
#         if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
#             assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
#         space = ' ' * (len(s) + 1)
#         for i, d in enumerate(devices):
#             p = torch.cuda.get_device_properties(i)
#             s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
#     else:
#         s += 'CPU\n'

#     if not newline:
#         s = s.rstrip()
#     # LOGGER.info(s.encode().decode('ascii', 'ignore') if system() == 'Windows' else s)  # emoji-safe
#     LOGGER.info(s.encode().decode('ascii', 'ignore')) # if system() == 'Windows' else s)  # emoji-safe
#     return torch.device('cuda:0' if cuda else 'cpu')

def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


class Ensemble():
    print("Ensemble")