from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
import pathlib
import os
import cv2
import numpy
from time import time as now
from PIL import Image
from payload import Payload

'''
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.resize_tensor_input(input_details[0]['index'],[154, 1, 50, 50])
interpreter.allocate_tensors()

interpreter.set_tensor(input_details[0]['index'], samples.astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predictions = [np.argmax(output_data[i]) for i in range(len(output_data))]
'''

class AIModel:
    __script_dir = pathlib.Path(__file__).parent.absolute()
    __model_file = os.path.join(__script_dir, '892.tflite')

    def __init__(self):
        self.interpreter = make_interpreter(self.__model_file)
        self.size = common.input_size(self.interpreter)
        self.interpreter.allocate_tensors()

        self.labels = ['black', 'amazon', 'clear', 'styro', 'null']

    def getPrediction(self, image, payload):
        sample = Image.fromarray(image).convert('L').resize(self.size, Image.ANTIALIAS)
        common.set_input(self.interpreter, sample)
        self.interpreter.invoke()
        classes = classify.get_classes(self.interpreter, top_k=1)
        prediction = classes[0].id
        payload.type = prediction

        return prediction
