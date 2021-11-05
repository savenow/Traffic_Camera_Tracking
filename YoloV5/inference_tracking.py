
import matplotlib
matplotlib.use('TKAgg')

import importlib
import sys

import torch
import time
import cv2
import numpy as np
import random

from Traffic_Camera_Tracking.YoloV5.yolo_v5_main_files.utils.torch_utils import time_sync
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from google.colab.patches import cv2_imshow

# # Importing sort
from Traffic_Camera_Tracking.SORT_Tracking.sort_yolo import Sort
# importlib.reload(Sort)

def show_tracker_bbox(input, frame, Score_ClassIDs):
    classID_dict = {0: ("Escooter", (0, 90, 255)), 1: ("Pedestrians", (255, 90, 0)), 2: ("Cyclists", (90, 255, 0))}
    
    img = frame
    for detection, Score_ClassID in zip(input, Score_ClassIDs):
        score = Score_ClassID[0]
        classID = int(Score_ClassID[1])
        color = classID_dict[classID][1]

        # For bounding box
        #print(f"Detection: {detection}")
        tracker_id = int(detection[4])
        x1, y1, x2, y2 = detection[0:4]
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        #print(f"Tracker ID: {tracker_id}, classID:{classID}")
        #print(f"X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}")

        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        label_1 = f'Track_ID: {tracker_id}'
        (w1, h1), _ = cv2.getTextSize(
                label_1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        label_2 = f'{classID_dict[classID][0]} {round(score*100, 1)}%'
        (w2, h2), _ = cv2.getTextSize(
                label_2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prints the text.    
        img = cv2.rectangle(img, (x1, y1 - 40), (x1 + w1, y1), color, -1)
        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w2, y1), color, -1)
        text_color = (0, 0, 0)
        img = cv2.putText(img, label_1, (x1, y1 - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        img = cv2.putText(img, label_2, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        del color
    return img


def show_tracker_bbox_2(input, frame):
    classID_dict = {0: ("Escooter", (0, 90, 255)), 1: ("Pedestrians", (255, 90, 0)), 2: ("Cyclists", (90, 255, 0))}
    #print(input)
    img = frame
    for detection in input:
        # For bounding box
        #print(f"Detection: {detection}")
        tracker_id = int(detection[9])
        x1, y1, x2, y2 = detection[0:4]
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        score = detection[4]
        classID = detection[5]

        #print(f"Tracker ID: {tracker_id}, classID:{classID}")
        #print(f"X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}")

        color = classID_dict[classID][1]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        label_1 = f'Track_ID: {tracker_id}'
        (w1, h1), _ = cv2.getTextSize(
                label_1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        label_2 = f'{classID_dict[classID][0]} {round(score*100, 1)}%'
        (w2, h2), _ = cv2.getTextSize(
                label_2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prints the text.    
        img = cv2.rectangle(img, (x1, y1 - 40), (x1 + w1, y1), color, -1)
        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w2, y1), color, -1)
        text_color = (0, 0, 0)
        img = cv2.putText(img, label_1, (x1, y1 - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        img = cv2.putText(img, label_2, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        del color
    return img

class TqdmExtraFormat(tqdm):
    """Provides a `total_time` format parameter"""
    @property
    def format_dict(self):
        d = super(TqdmExtraFormat, self).format_dict
        total_time = d["elapsed"] * (d["total"] or 0) / max(d["n"], 1)
        d.update(total_time=self.format_interval(total_time) + " in total")
        return d


def video_inference(input_path, output_path, model_weights, class_name=None):
    video_capture = cv2.VideoCapture(input_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    codec = 'mp4v'

    video_output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), float(fps), (width, height),)
    model = torch.hub.load(r'/content/Traffic_Camera_Tracking/YoloV5/yolo_v5_main_files',
                           'custom',
                           path=model_weights,
                           source='local')
    framecount = 1

    device = torch.device('cuda:0')
    cudnn.benchmark = True
   
    #del Object_tracker
    Object_tracker = Sort(max_age=30, min_hits=7, iou_threshold=0.15)
    Object_tracker.reset_count()

    pbar = TqdmExtraFormat(total = total_frames, desc='Inference Progress: ')
    while video_capture.isOpened():
        _, frame = video_capture.read()
        
        if _:
            # Results. Try printing or modifying this array to get different classes
            results = model(frame)
            result = results.xyxy[0]
            
            score_class_id = []
            if len(result) > 0:  
              dets = []
              for items in result:
                # items[:5].detach().cpu().numpy()
                #print(items)
                dets.append(items[:].tolist())
                #score_class_id.append((items[4].item(), items[5].item()))

              dets = np.array(dets)
              #print(dets)
              tracker = Object_tracker.update(dets)
            else:
              tracker = Object_tracker.update()
            
            video_output.write(show_tracker_bbox_2(tracker, frame))
            pbar.update(framecount)
            #print(Object_tracker)
        else:
          break

    video_capture.release()
    video_output.release()

# Model
model_weight = r'/content/tl_extended_epoch_30.pt'
input_directory = r'/content/drive/MyDrive/YOLO/Sample Videos/08-06-2021_09-40.mkv'
output_directory = r'/content/drive/MyDrive/YOLO/tl_extended_results/08-06-2021_09-40.mkv'
class_name = 'Escooter'
video_inference(input_directory, output_directory, model_weight, class_name)
