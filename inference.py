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
from utils.datasets import LoadImages
from utils.torch_utils import time_sync
from utils.general import non_max_suppression, scale_coords, check_img_size
from hubconf import custom

from sort_yoloV5 import Sort
from visualizer import Visualizer
from calibration import Calibration

class Inference():
    def __init__(self, input, model_weights, output=None, minimap=False, imgSize=[1408, 1408]):        
        # Inference Params
        self.img_size = imgSize
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.max_det = 1000
        self.classes = None # Filter classes

        self.device = torch.device('cuda:0')
        self.half = True
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
                self.fps = 30
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
        
        # Loading Model
        model = DetectMultiBackend(self.model_weights, device=self.device, dnn=None)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        self.names = ['Escooter', 'Pedestrian', 'Cyclist']
        
        if self.pt:
            model = model.model.half()
        self.model = model

        # Initialize Tracker
        self.Objtracker = Sort(max_age=30, min_hits=7, iou_threshold=0.15)
        self.Objtracker.reset_count()

        # Camera Calibration data: Used for velocity estimation
        self.enable_minimap = minimap
        self.Calib = Calibration()
        
        # Parameters for velocity estimation
        self.velocity_frame_window = 5
        self.trackDict = defaultdict(list)
        self.trackCount = 0

        self.runInference()
    
    def UpdateTracker(self, pred):
        if len(pred) > 0:
            dets = []
            for items in pred:
                dets.append(items[:].tolist())
            self.trackCount += 1
        
            dets = np.array(dets)
            self.tracker = self.Objtracker.update(dets)
        else:
            self.tracker = self.Objtracker.update()
        
    def VelocityEstimation(self, velocity_array):    
        for detection in self.tracker:
            center_x = (detection[0] + detection[1])/2
            center_y = (detection[2] + detection[3])/2
            

            trackID = int(detection[9])
            self.trackDict[trackID].append((center_x, center_y))
            
            if len(self.trackDict[trackID]) > 2: 
                
                previous_point = self.Calib.projection_pixel_to_world(self.trackDict[trackID][-2])
                current_point = self.Calib.projection_pixel_to_world(self.trackDict[trackID][-1])

                del self.trackDict[trackID][-2]

                distance_metres = round(float(math.sqrt(math.pow(previous_point[0] - current_point[0], 2) + math.pow(previous_point[1] - current_point[1], 2))), 2)
                speed_kmH = round(float((distance_metres * self.fps)/2) * 3.6 , 2)
                output_array = np.append(detection, speed_kmH)
                velocity_array.append(output_array)
            
        return velocity_array

    def runInference(self):
        dataset = LoadImages(self.input, img_size=self.img_size, stride=self.stride, auto=self.pt and not self.jit)
        bs = 1
        vid_path, vid_writer = [None] * bs, [None] * bs

        Visualize = Visualizer(self.enable_minimap)
        dt, seen = [0.0, 0.0, 0.0], 0
        framecount = 0
        for path, im, im0, vid_cap, s in dataset:
            framecount += 1
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = self.model(im, augment=False, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2
            if self.pt:
                pred = pred[0]

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                s += '%gx%g ' % im.shape[2:] 

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                print(f'{s}Done. ({1/(t3 - t2):.3f}fps)')
            
            # Save the images or videos
            if self.inference_mode == 'SingleImage':
                self.frame = Visualize.drawBBOX(pred[0], im0)
                cv2.imwrite(self.output, self.frame)
            
            elif self.inference_mode == 'Video':
                # Update the tracker
                self.UpdateTracker(pred[0])
                
                # Velocity Estimation
                velocity_estimation = []
                calculated_velocity = self.VelocityEstimation(velocity_estimation)
                
                if calculated_velocity:
                    self.frame = Visualize.drawAll(calculated_velocity, im0)
                elif len(self.tracker) > 0:
                    self.frame = Visualize.drawTracker(self.tracker, im0)
                elif len(pred) > 0:
                    self.frame = Visualize.drawBBOX(pred[0], im0)
                                
                if vid_path[i] != self.output:  # new video
                    vid_path[i] = self.output
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  
                        w, h = im0.shape[1], im0.shape[0]
                        
                    vid_writer[i] = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
                vid_writer[i].write(self.frame) 
            
            if framecount > 300:
                vid_writer[i].release()
                break
            
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.img_size)}' % t)
    
if __name__ == "__main__":
    Inference(
        '/content/31.mp4', 
        '/content/tl_l6_89k_bs24_im1408_e150.pt',
        '/content/31_minimap.mkv',
        minimap=True
    )