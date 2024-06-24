# # **Yolov5 Object Detection - NYC**

# #### **Motivation:** Project that utilizes YoloV5's pre-trained CNN to detect objects within a video in NYC (Timelapse footage of people walking inside Grand Central Station in New York.)

!git clone https://github.com/ultralytics/yolov5
%cd yolov5

# Installing Necessary Dependencies within the yolov5 .txt dependencies list
!pip install -r requirements.txt

import sys
sys.path.append('/kaggle/working/yolov5')
import os
print(os.listdir('utils'))

### Main Dependencies
import cv2
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Processing devices selected for processing, I'm on Kaggle, using Tesla T4 X2 GPU's, hence '(0,1)' since I'm using 2 GPU's
device = select_device('0,1')

# Loading in the pre-trained YOLOv5 model
model = DetectMultiBackend('yolov5s.pt', device=device) #yolov5.pt links to the yolov5 pre-trained model weights (params)
stride, names, pt = model.stride, model.names, model.pt

# Opening the video file
video_path = '/kaggle/input/nyc-video/coverr-timelapse-of-grand-central-station-7681-1080p.mp4'  # Replace with your video path
dataset = LoadImages(video_path, img_size=640, stride=stride)

# Defining the fourcc codec and creating the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Retrieving the frame width and height from the video capture
vid_cap = cv2.VideoCapture(video_path)
frame_width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
vid_cap.release()

out = cv2.VideoWriter('/kaggle/working/output_video.mp4', fourcc, 30.0, (frame_width, frame_height))

# Looping through the video frames
# path = path to current video frame
# img = the current video frame/img(image) as a numpy array, preprocessed for the yolov5 model
# im0s = the original frame as a numpy array, planned to be used for displaying the results
# vid_cap = video capture object for accessing video properties
# s = string used for displaying processing information
for path, img, im0s, vid_cap, s in dataset:
    img = torch.from_numpy(img).to(device) #converting the video frame from a numpy array to a PyTorch tensor, moving the tensor to the specified device (GPU)
    img = img.float() / 255.0  # Normalizing the image pixel values to the range of [0, 1], converting the tensor to floating-point numbers
    if len(img.shape) == 3:
        img = img[None]  # Adding batch dimension

    # Inference
    pred = model(img, augment=False, visualize=False) #disabling augmentations and visualization

    #Applying NMS (non-max suppression) = NMS is a function that filters out overlapping bounding boxes, keeping only the most confident ones
    #pred = raw predictions from the model
        #0.25 = Confidence Threshold
        #0.45 = IoU (Intersection over Union threshold for NMS, overlapping boxes with IoU above this value are suppressed)
        #None = no class-specific NMS threshold used, False = Whether to apply NMS for multi-class
        #max_det=1000, max number of detections to keep
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    # Processing frame detections
    for i, det in enumerate(pred):  #iterating over a list of predictions (one for each image in the batch)
        im0 = im0s.copy() #copying the original frame to annotate it with decision results
        annotator = Annotator(im0, line_width=2, example=str(names)) #im0 = frame to annotate, line_width=2 is the width of the bounding box lines, example=str(names) provides a string example of class names
        if len(det): #checking for detections, scale_boxes = adjusting bounding box coords, img.shape[2:] are the dimensions of the model input, det[:, :4] are the bounding box coordinates in the model output, im0.shape retrieves the dimensions of the original frame, round(), rounds the coordinates to integer values for drawing
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

            # Writing results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(cls))

        # Ensuring frame size matches
        frame = cv2.resize(im0, (frame_width, frame_height))

        # Writing frame to video
        out.write(frame)

# Releasing everything once the processing is complete
out.release()