# from yolov5 GitHub
# Used for loading model for main_kinect.py

import torch
import torch.nn as nn

from experimental_kinect import attempt_load

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

    