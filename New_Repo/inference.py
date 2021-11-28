import torch
import cv2
import numpy as np
import os
import math
from torch._C import device
from tqdm import tqdm
from collections import namedtuple, defaultdict
import torch.backends.cudnn as cudnn
import sys
sys.path.append('./yolo_v5_main_files')

from models.common import DetectMultiBackend, AutoShape
from utils.torch_utils import time_sync
from utils.general import non_max_suppression, scale_coords, check_img_size
from hubconf import custom

from sort_yoloV5 import Sort
from visualizer import Visualizer
from calibration import Calibration

class Inference():
    def __init__(self, input, model_weights, output=None, imgSize=1408):        
        # Inference Params
        self.img_size = imgSize
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.max_det = 1000
        self.classes = None # Filter classes

        self.device = torch.device('cuda:0')
        
        cudnn.benchmark = True
        
        
        # Checking input
        if os.path.isfile(input):
            # Further functionality needs to be added for Folder Inference :))
            if input[-4:] in ['.png', '.jpg']:
               self.input = input
               self.inference_mode = 'SingleImage'
            elif input[-4:] in ['.mp4', '.mkv', '.avi']:
                self.input = input
                self.inference_mode = 'Video'
            else:
                print("Invalid input file. The file should be an image or a video !!")
                exit(-1)
        else:
            print("Input file doesn't exist. Check the input path")
            exit(-1)
                 
        # Checking weights file
        if os.path.isfile(model_weights):
            if model_weights[-3:] == '.pt':
                self.model_weights = model_weights
                self.inference_backend = 'PyTorch'
            elif model_weights[-7:] == '.engine':
                self.model_weights = model_weights
                self.inference_backend = 'TensorRT'
            else:
                print(f"Invalid Weights file. {model_weights} does not end with '.engine' or '.pt'")
                exit(-1)
        else:
            print("Model weights file does not exist. Check the weights path")
            exit(-1)
        
        # Checking output
        if output == None:
            self.output = self.input.split('/')[-1]
        else:
            self.output = output
        
        if self.inference_backend == 'PyTorch':
            self.model = torch.hub.load('./yolo_v5_main_files',
                            'custom',
                            path=self.model_weights,
                            source='local')
        else:
            self.model = DetectMultiBackend(self.model_weights, device=self.device, dnn=False)
            self.model = AutoShape(self.model)

        # Initialize Tracker
        self.Objtracker = Sort(max_age=30, min_hits=7, iou_threshold=0.15)
        self.Objtracker.reset_count()

        # Camera Calibration data: Used for velocity estimation
        self.Calib = Calibration()
        
        # Parameters for velocity estimation
        self.velocity_frame_window = 5
        self.trackDict = defaultdict(list)
        self.trackCount = 0

        # Running inference on different types of input
        if self.inference_mode == 'Video':
            self.VideoInference()
    
    def InferFrame(self):
        #if self.inference_backend == 'PyTorch':
        results = self.model(self.frame, size=self.img_size)
        return results.xyxy[0]
       

    def VideoInference(self):
        video_capture = cv2.VideoCapture(self.input)

        self.width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / self.fps
        codec = 'mp4v'

        video_output = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*codec), float(self.fps), (self.width, self.height),)

        Visualize = Visualizer()
        frame_count = 0
        while video_capture.isOpened():
            _, self.frame = video_capture.read()

            if _:
                frame_count += 1
                output = self.InferFrame()
                #print(output)
                # Updating the tracker
                if len(output) > 0:
                    dets = []
                    for items in output:
                        dets.append(items[:].tolist())
                    self.trackCount += 1
                
                    dets = np.array(dets)
                    tracker = self.Objtracker.update(dets)
                else:
                    tracker = self.Objtracker.update()
                
                # Values for velocity estimation
                # for detection in tracker:
                #     center_x = (detection[0] + detection[1])/2
                #     center_y = (detection[2] + detection[3])/2
                #     self.trackDict[int(detection[9])].append((center_x, center_y))
                #     self.trackCount += 1
                
                velocity_estimation = []
                # Velocity Estimation
                #if self.velocity_frame_window < self.trackCount:
                    # for i in self.trackDict.keys():
                    #     if len(self.trackDict[i]) > self.velocity_frame_window:
                    #         previous_point = self.Calib.projection_pixel_to_world(self.trackDict[i][0])
                    #         current_point = self.Calib.projection_pixel_to_world(self.trackDict[i][-1])

                    #         del self.trackDict[i][0]

                    #         distance_metres = round(float(math.sqrt(math.pow(previous_point[0] - current_point[0], 2) + math.pow(previous_point[1] - current_point[1], 2))), 2)
                    #         speed_kmH = round(float((distance_metres * self.fps)/ (self.velocity_frame_window)) * 3.6 , 2)
                    
                for detection in tracker:
                    center_x = (detection[0] + detection[1])/2
                    center_y = (detection[2] + detection[3])/2
                    

                    trackID = int(detection[9])
                    self.trackDict[trackID].append((center_x, center_y))
                    #self.trackCount += 1
                    
                    if len(self.trackDict[trackID]) > 2: #self.velocity_frame_window:
                        
                        previous_point = self.Calib.projection_pixel_to_world(self.trackDict[trackID][0])
                        current_point = self.Calib.projection_pixel_to_world(self.trackDict[trackID][1])

                        del self.trackDict[trackID][0]

                        distance_metres = round(float(math.sqrt(math.pow(previous_point[0] - current_point[0], 2) + math.pow(previous_point[1] - current_point[1], 2))), 2)
                        speed_kmH = round(float((distance_metres * self.fps)) * 3.6 , 2)
                        #print(previous_point)
                        #print(current_point)
                        # print(distance_metres)
                        # print(speed_kmH)
                        velocity_estimation.append(np.append(detection, speed_kmH))
                
                
                if velocity_estimation:
                    self.frame = Visualize.drawAll(velocity_estimation, self.frame)
                elif len(tracker) > 0:
                    self.frame = Visualize.drawTracker(tracker, self.frame)
                elif len(output) > 0:
                    self.frame = Visualize.drawBBOX(output, self.frame)
                
                video_output.write(self.frame)
            else:
                break
        
        video_capture.release()
        video_output.release()

    
    
if __name__ == "__main__":
    Inference(
        '/media/mydisk/videos/input_new/29/29.mp4', 
        '/home/students-fkk/Traffic_Camera_Tracking/tl_l6_89k_bs24_im1408_e150.engine',
        '/media/mydisk/videos/output_e150/29.mp4'
    )