# from pycoral.adapters import common
# from pycoral.adapters import classify
# from pycoral.utils.edgetpu import make_interpreter
import pathlib
import os
import cv2
# import numpy
import time
# from time import time as now
# from PIL import Image
import torch
from payload import Payload
# from segmentation2_kinect import getPayload
# import Yolov5Model
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

rgb_image = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
rgb_image = cv2.resize(rgb_image, (640, 480)) # (192, 224)

def main():
    # payload = getPayload(rgb_image)
    # predicted_labels = Yolov5Model.getPrediction(Yolov5Model.model, rgb_image, payload)
    # print("predicted_labels:", predicted_labels)
    print("main")


class Yolov5Model:
    __script_dir = pathlib.Path(__file__).parent.absolute()
    # __model_file = os.path.join(__script_dir, '892.tflite')

    # TODO: Look into converting from pytorch to onnx to tflite
    def __init__(self):
        # TODO: Figure out what to do here
        # FIXME: Ask Tim: Is this loading the model?
        # self.interpreter = make_interpreter(self.__model_file)
        # self.size = common.input_size(self.interpreter)
        # self.interpreter.allocate_tensors()

        # TODO: Load the model here?
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='ml_files/best.pt')  # local model

        # # Set custom inference settings
        # self.model.agnostic = False            # NMS class-agnostic - Model uses the foreground to create bounding boxes instead of classes (pre-processor)
        # # self.model.amp = False               # Automatic Mixed Precision (AMP) inference - If True, Applicable calculations are computed in 16-bit precision instead of 32-bit precision
        # self.model.amp = True                  # Speeds up inference process
        # self.model.classes = None              # (optional list) Filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        # self.model.conf = 0.60                 # NMS confidence threshold - Confidence value that the bounding box contains an object (Max = 1)
        # self.model.dmb = True                  # 
        # self.model.dump_patches = False        # 
        # self.model.iou = 0.45                  # NMS IoU threshold - Bounding box overlap threshold
        # self.model.max_det = 6                 # Maximum number of detections per image
        # self.model.multi_label = False         # NMS multiple labels per box
        # self.model.names = ['black_box', 'orange_bucket', 'styrofoam_box', 'null']         # Names of containers
        # self.model.pt = True                   # Use ML model weights that are the .pt format
        # self.model.stride = 32                 # 
        # self.model.training = True             # 

        # self.labels = ['black', 'amazon', 'clear', 'styro', 'null']
        # self.labels = ['black_box', 'styrofoam_box', 'orange_bucket', 'null']
        self.labels = ['black_box', 'orange_bucket', 'styrofoam_box', 'null']

    # def getPrediction(self, model, image, payload):
    def getPrediction(self, model, image, payload):
        # sample = Image.fromarray(image).convert('L').resize(self.size, Image.ANTIALIAS)
        # common.set_input(self.interpreter, sample)
        # self.interpreter.invoke()
        # classes = classify.get_classes(self.interpreter, top_k=1)
        # prediction = classes[0].id
        # payload.type = prediction
        # model = torch.hub.load('ultralytics/yolov5', 'custom', path='ml_files/best.pt')  # local model

        image = cv2.resize(image, (224, 192))
        # image = cv2.resize(image, (640, 480))

        # Stuff for FPS text on image
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # origin = (5, 15)
        # font_scale = 0.5
        # color = (255, 0, 0) # BGR
        # thickness = 2

        # Set start time for FPS calculations
        start_time = time.time()

        # Choose custom inference size
        results = model(image, size=224) # 640, 416, 360, 224, 160

        # Save the arrays of labels and the bounding box coordinates
        predicted_labels = results.xyxyn[0][:, -1].numpy()
        cord_thres = results.xyxyn[0][:, :-1].numpy()

        print("predicted_labels:", predicted_labels)

        # Display the results in the terminal
        print("\nConcise Inference Results:")
        # Print results
        results.print()

        # print("\nCustom Inference Results:")

        # Calculate the FPS
        fps = '{:.1f}'.format(1 / (time.time() - start_time))

        # Display FPS in the terminal
        print(f'FPS: {fps}\n')

        # Get prediction/s
        if len(predicted_labels) != 0:
            # TODO: This may just print the 1st element every time
            # TODO: Make sure payload.type is correct
            # Convert predicted class labels (numeric) to readable labels
            # for i in range(len(predicted_labels)):
            #     payload = predicted_labels[i]
            #     if predicted_labels[i] == 0:
            #         print("Black, plastic box was found!")
            #         # return predicted_labels[i]
            #     elif predicted_labels[i] == 1:
            #         print("Orange, plastic bucket was found!")
            #         # return predicted_labels[i]
            #     elif predicted_labels[i] == 2:
            #         print("White, Styrofoam box was found!")
            #         # return predicted_labels[i]
            #     else:
            #         print("An error occurred!")
            return predicted_labels
        else:
            print("No containers were found!")
            return 3


        # Calculate the FPS
        # fps = 'FPS: {:.1f}'.format(1 / (time.time() - start_time))

        # Display FPS in the terminal
        # print(f'FPS: {fps}\n')

        # TODO: Maybe sort/order predictions as a list (first available index = black box, etc. )
        # What was its type before?
        # return prediction
        # return predicted_labels

if __name__ == "__main__":
    main()