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

#from sort_yoloV5 import Sort
from visualizer import Visualizer
from calibration import Calibration

class Inference():
    def __init__(self, input, model_weights, output=None, imgSize=[1408, 1408]):        
        # Inference Params
        self.img_size = imgSize
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.max_det = 1000
        self.classes = None # Filter classes

        self.device = torch.device('cuda:0')
        self.half = True # FP16 Inference
        cudnn.benchmark = True
        
        self.input = input
        self.model_weights = model_weights
        self.output = output

        # Loading Model
        model = DetectMultiBackend(self.model_weights, device=self.device, dnn=None)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        self.names = ['Escooter', 'Pedestrian', 'Cyclist']
        
        if self.pt:
            model = model.model.half()
        self.model = model
        
        print(self.runInference()[0])

    def runInference(self):
        dataset = LoadImages(self.input, img_size=self.img_size, stride=self.stride, auto=self.pt and not self.jit)
        bs = 1
        
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0, vid_cap, s in dataset:
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

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *self.img_size)}' % t)
        return pred
        

if __name__ == "__main__":
    Inference(
        '/home/students-fkk/Traffic_Camera_Tracking/Sample_Pictures/1.png', 
        '/home/students-fkk/Traffic_Camera_Tracking/tl_l6_89k_bs24_im1408_e150.engine',
        '/home/students-fkk/Pictures/1_test.png'
    )