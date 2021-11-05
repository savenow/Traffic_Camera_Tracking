import matplotlib
matplotlib.use('TKAgg')

import torch
import cv2
import numpy as np
from tqdm import tqdm
import torch.backends.cudnn as cudnn

# Importing custom sort
from sort_yolo import Sort


def show_tracker_bbox(input, frame):
    classID_dict = {0: ("Escooter", (0, 90, 255)), 1: ("Pedestrian", (255, 90, 0)), 2: ("Cyclist", (90, 255, 0))}
    
    img = frame
    for detection in input:
        # Extracting data from the detection
        tracker_id = int(detection[9])
        x1, y1, x2, y2 = detection[0:4]
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        score = detection[4]
        classID = detection[5]

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
    # model = torch.hub.load(r'YoloV5\yolo_v5_main_files',
    #                        'custom',
    #                        path=model_weights,
    #                        source='local')
    model = torch.jit.load(model_weights)
    model.eval()
    torch.jit.optimize_for_inference(model)
    model.cuda()
    model.half()

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
            #frame = frame[None]
            frame = cv2.resize(frame, (1920, 1920))
            frame = torch.from_numpy(frame).to(device)
            frame = frame.half()
            frame /= 255
            if len(frame.shape) == 3:
              frame = frame[None]
            print(frame)
            results = model(frame)
            result = results.xyxy[0]
            
            if len(result) > 0:  
              dets = []
              for items in result:
                dets.append(items[:].tolist())
                
              dets = np.array(dets)
              tracker = Object_tracker.update(dets)
            else:
              tracker = Object_tracker.update()
            
            video_output.write(show_tracker_bbox(tracker, frame))
            pbar.update(framecount)

        else:
            break

    video_capture.release()
    video_output.release()

# Model
model_weight = r'D:\Vishal-Videos\Project_Escooter_Tracking\tl_extended_epoch_30.torchscript.pt'
input_directory = r'D:\Vishal-Videos\Project_Escooter_Tracking\input_new\50\50.mp4'
output_directory = r'D:\Vishal-Videos\Project_Escooter_Tracking\samples\infered\50_tracking.mkv'
class_name = 'Escooter'
video_inference(input_directory, output_directory, model_weight, class_name)
