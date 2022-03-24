# Load and Test/Inference the ML Model on an image (Custom detect.py)
import torch
import time
import cv2

# Load the ML model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='ml_files/best.pt')  # local model

# Set custom inference settings
model.conf = 0.60           # NMS confidence threshold - Confidence value that the bounding box contains an object (Max = 1)
iou = 0.45                  # NMS IoU threshold - Bounding box overlap threshold
agnostic = False            # NMS class-agnostic - Model uses the foreground to create bounding boxes instead of classes (pre-processor)
multi_label = False         # NMS multiple labels per box
classes = None              # (optional list) Filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
max_det = 6                 # Maximum number of detections per image
# amp = False               # Automatic Mixed Precision (AMP) inference - If True, Applicable calculations are computed in 16-bit precision instead of 32-bit precision
amp = True                  # Speeds up inference process

# Choose test image
image = cv2.imread('three_containers1.jpg')[..., ::-1]
image = cv2.resize(image, (224, 192))

# Name of video feed window
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
results = model(image, size=224) # 640, 416, 224, 360, 160

# Save the arrays of labels and the bounding box coordinates
predicted_labels = results.xyxyn[0][:, -1].numpy()
cord_thres = results.xyxyn[0][:, :-1].numpy()

# Display the results in the terminal
print("\nConcise Inference Results:")
results.print()
print("\nCustom Inference Results:")

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
text = 'FPS: {:.1f}'.format(1 / (time.time() - start_time))

# Display FPS on the video feed
image = cv2.putText(image, text, origin, font, font_scale, color, thickness, cv2.LINE_AA)
cv2.imshow(video_feed_name, image)

# Display FPS in the terminal
print(text, '\n')

# Press "any" button to end the program (I normally use spacebar.)
cv2.waitKey(0)
