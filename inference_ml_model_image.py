# Load and Test/Inference the ML Model on an image (Custom detect.py)
import torch
import time
import cv2

# Load the ML model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='mtre4800-kawasaki-project/best.pt')  # local model

# Set custom inference settings
model.agnostic = False            # NMS class-agnostic - Model uses the foreground to create bounding boxes instead of classes (pre-processor)
# model.amp = False               # Automatic Mixed Precision (AMP) inference - If True, Applicable calculations are computed in 16-bit precision instead of 32-bit precision
model.amp = True                  # Speeds up inference process
model.classes = None              # (optional list) Filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.conf = 0.60                 # NMS confidence threshold - Confidence value that the bounding box contains an object (Max = 1)
model.dmb = True                  #
model.dump_patches = False        #
model.iou = 0.45                  # NMS IoU threshold - Bounding box overlap threshold
model.max_det = 6                 # Maximum number of detections per image
model.multi_label = False         # NMS multiple labels per box
model.names = ['black_box', 'orange_bucket', 'styrofoam_box']         # Names of containers
model.pt = True                   # Use ML model weights that are .pt format
model.stride = 32                 #
model.training = False            #

# Choose and resize test image
image = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')[..., ::-1]
# image = cv2.imread('mtre4800-kawasaki-project/three_containers1.jpg')
image = cv2.resize(image, (640, 480))

# Name the video feed window
video_feed_name = "Video Feed"
cv2.namedWindow(video_feed_name, cv2.WINDOW_NORMAL)

# Stuff for FPS text
font = cv2.FONT_HERSHEY_SIMPLEX
origin = (5, 15)
font_scale = 0.5
color = (255, 0, 0) # BGR
thickness = 2

# Set start time for FPS calculations
start_time = time.time()

# Choose custom inference size and save test results
results = model(image, size=640) # 640, 416, 224, 360, 160

# Save the arrays of labels and the bounding box coordinates
predicted_labels = results.xyxyn[0][:, -1].numpy()
cord_thres = results.xyxyn[0][:, :-1].numpy()

print("predicted_labels:", predicted_labels)

# Display the results in the terminal
print("\nConcise Inference Results:")
results.print()
print("\nCustom Inference Results:")

# Get prediction
if len(predicted_labels) != 0:
    # Convert predicted class labels (numeric) to readable labels
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == 0:
            print("Black, plastic box was found!")
        elif predicted_labels[i] == 1:
            print("Orange, plastic bucket was found!")
        elif predicted_labels[i] == 2:
            print("White, Styrofoam box was found!")
        else:
            print("An error occurred!")
else:
    print("No containers were found!")

# Calculate the FPS
fps = '{:.1f}'.format(1 / (time.time() - start_time))

# Display FPS on the video feed
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.putText(image, fps, origin, font, font_scale, color, thickness, cv2.LINE_AA)
cv2.imshow(video_feed_name, image)

# Display FPS in the terminal
print(f'FPS: {fps}\n')

# Press "any" button to end the program (I normally use spacebar.)
cv2.waitKey(0)


# Draw bounding boxes
# def drawBoundingBox(color, predicted_labels, cord_thres):
def drawBoundingBox(color, frame, predicted_labels, cord_thres):
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(len(predicted_labels)):
        row = cord_thres[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # cv2.putText(frame, predicted_labels[i], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
