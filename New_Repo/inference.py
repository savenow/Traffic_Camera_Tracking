import torch
import cv2
import numpy as np
import os
import math
from tqdm import tqdm
from collections import namedtuple, defaultdict
import torch.backends.cudnn as cudnn

from yolo_v5_main_files.models.common import DetectMultiBackend
from yolo_v5_main_files.utils.torch_utils import time_sync
from yolo_v5_main_files.utils.general import non_max_suppression, scale_coords

from sort_yoloV5 import Sort
from visualizer import Visualizer
from calibration import Calibration

class Inference():
    def __init__(self, input, model_weights, output=None, imgSize=1408):        
        # Inference Params
        self.img_size = imgSize
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.max_det = 1000
        self.classes = None # Filter classes

        device = torch.device('cuda:0')
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
        
        
        # Load models
        # if self.inference_backend == 'PyTorch':
        #     self.model = torch.hub.load(
        #         'yolo_v5_main_files',
        #         'custom',
        #         path=self.model_weights,
        #         source='local'
        #     )
        #     self.model.eval()
        #     self.model.cuda()
        #     self.model.half()

        # else:
        #     try:
        #         import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
        #     except ImportError as e:
        #         print(f"TensorRT not installed -> Error :{e}")
        #         exit(-1)
        
        #     Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        #     logger = trt.Logger(trt.Logger.INFO)
        #     with open(self.model_weights, 'rb') as f, trt.Runtime(logger) as runtime:
        #         model = runtime.deserialize_cuda_engine(f.read())
        #     self.bindings = dict()
        #     for index in range(model.num_bindings):
        #         name = model.get_binding_name(index)
        #         dtype = trt.nptype(model.get_binding_dtype(index))
        #         shape = tuple(model.get_binding_shape(index))
        #         #print(f"Shape: {shape}")
        #         data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        #         self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        #     self.binding_addrs = {n: d.ptr for n, d in self.bindings.items()}
        #     self.context = model.create_execution_context()
        #     self.batch_size = self.bindings['images'].shape[0]
        #     #print(self.batch_size)
        model = DetectMultiBackend(self.model_weights, device=device)
        stride, self.names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        #imgsz = check_img_size(imgsz, s=stride)  # check image size

        # FP16(Half) Inference -> Only on CUDA Cores
        half = (pt or engine) and device.type != 'cpu'
        if pt:
            model.model.half() if half else model.model.float()
        # Warmup check
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *self.img_size).to(device).type_as(next(model.model.parameters())))

        self.model = model

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
        #    results = self.model(self.frame, size=self.img_size)
        #    return results.xyxy[0]
        # Preprocess data
        
        im = torch.from_numpy(self.frame).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        
        # Inference
        pred = self.model(im)
        t3 = time_sync()
        
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        
        output_string = ''
        # Process Predictions
        for i, det in enumerate(pred):
            output_string +=  '%gx%g ' % im.shape[2:]

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    output_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
        print(f'{output_string}Done. ({1/(t3 - t2):.3f}fps)')

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
        
        while video_capture.isOpened():
            _, self.frame = video_capture.read()

            if _:
                output = self.InferFrame()

                # Updating the tracker
                if len(output) > 0:
                    dets = []
                    for items in output:
                        dets.append(items[:].tolist())
                
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
                
                # Velocity Estimation
                if self.velocity_frame_window < self.trackCount:
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
                        self.trackCount += 1

                        if len(self.trackDict[trackID]) > self.velocity_frame_window:
                            previous_point = self.Calib.projection_pixel_to_world(self.trackDict[i][0])
                            current_point = self.Calib.projection_pixel_to_world(self.trackDict[i][-1])

                            del self.trackDict[trackID][0]

                            distance_metres = round(float(math.sqrt(math.pow(previous_point[0] - current_point[0], 2) + math.pow(previous_point[1] - current_point[1], 2))), 2)
                            speed_kmH = round(float((distance_metres * self.fps)/ (self.velocity_frame_window)) * 3.6 , 2)
                            detection.append(speed_kmH)
                

                self.frame = Visualize.drawAll(tracker, self.frame)

                video_output.write(self.frame)
            else:
                break
        
        video_capture.release()
        video_output.release()

    
    
if __name__ == "__main__":
    Inference(
        '/media/mydisk/videos/input_new/29/29.mp4', 
        '/home/students-fkk/Traffic_Camera_Tracking/tl_l6_89k_bs24_im1408_e150.pt',
        '/media/mydisk/videos/output_e150/29.mp4'
    )