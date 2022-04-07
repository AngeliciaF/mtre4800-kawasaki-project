# Load and Test/Inference the ML Model on a live video feed (Custom detect.py)
# python yolov5\detect.py --source 0 --img 416 --conf-thres 0.40 --weights best.pt

from ast import match_case
import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm
import seaborn as sn
import time
import cv2
import numpy as np
# from inference_ml_model_video import drawBoundingBox

# Load the ML model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='ml_files/best.pt')  # local model

# Custom inference settings
model.conf = 0.40           # NMS confidence threshold - Confidence value that the bounding box contains an object (Max = 1)
model.iou = 0.45                  # NMS IoU threshold - Bounding box overlap threshold
model.agnostic = False            # NMS class-agnostic - Model uses the foreground to create bounding boxes instead of classes (pre-processor)
model.multi_label = False         # NMS multiple labels per box
model.classes = None              # (optional list) Filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 6                 # Maximum number of detections per image
# amp = False               # Automatic Mixed Precision (AMP) inference - Applicable calculations are computed in 16-bit precision instead of 32-bit precision
model.amp = True                  # Speeds up inference process


# Name video feed window
# video_feed_name = "Video Feed"
# cv2.namedWindow(video_feed_name, cv2.WINDOW_NORMAL)

# Stuff for FPS text
font = cv2.FONT_HERSHEY_SIMPLEX
origin = (5, 15)
font_scale = 0.5
color = (255, 0, 0) # BGR
thickness = 2

# Choose camera
cap = cv2.VideoCapture(0)

while True:
    # Set start time for FPS calculations
    start_time = time.time()

    # Read from camera
    ret, frame = cap.read()
        
    if frame is None:
        break

    x_shape, y_shape = frame.shape[1], frame.shape[0]
    
    # print(frame.shape)
    # frame = cv2.resize(frame, (192,224))
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Choose custom inference size and save test results
    results = model(frame, size=224) # 640, 416, 224, 360, 160

    # Save the array of labels and the bounding box coordinates
    predicted_labels = results.xyxyn[0][:, -1].numpy()
    cord_thres = results.xyxyn[0][:, :-1].numpy()

    print("Predicted Labels:\n", predicted_labels)
    print("Bounding Boxes Coordinates:\n", cord_thres)

    # TODO: Use reference angle
    # Change bbox orientation/angle
    # Figure out scada tags stuff
    if len(predicted_labels) != 0:
        # Convert predicted class labels (numeric) to readable labels
        for i in range(len(predicted_labels)):
            row = cord_thres[i]
            # If the confidence is > 0.6, predict_labels
            if row[4] >= 0.6:
                if predicted_labels[i] == 0:
                    color = (255, 0, 0)
                    print("Black, plastic box was found!\n")
                elif predicted_labels[i] == 1:
                    color = (0, 255, 0)
                    print("Orange, plastic bucket was found!\n")
                elif predicted_labels[i] == 2:
                    color = (0, 0, 255)
                    print("White, Styrofoam box was found!\n")
                else:
                    print("An error occurred!\n")
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        print("No containers were found!\n")

    # Display FPS on the video feed and in the terminal
    text = 'FPS: {:.1f}'.format(1 / (time.time() - start_time))
    frame = cv2.putText(frame, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)
    print(text)

    cv2.waitKey(0)

    # Display bounding boxes
    # frame = cv2.rectangle(frame, (0,0), (100,100), color, thickness)
    # (cord_thres[0][0], cord_thres[0][1]), (cord_thres[0][2], cord_thres[0][3])

    # Display frame and convert from BGR to RGB
    # results.show()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Frame:", frame)

    if (cv2.waitKey(1) & 0xFF) == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Draw bounding boxes
# def drawBoundingBox(color, predicted_labels, cord_thres):
def drawBoundingBox(color, frame, predicted_labels, cord_thres):
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(len(predicted_labels)):
        row = cord_thres[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            # color = (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, predicted_labels[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
