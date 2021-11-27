import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from collections import namedtuple
import torch.backends.cudnn as cudnn
from visualizer import Visualizer
from sort_yoloV5 import Sort

class Inference():
    def __init__(self, input, model_weights, output=None, imgSize=1408):        
        # Checking input
        if os.path.isfile(input):
            if input[-4:] in ['.png', '.jpg']:
               self.input = input
               self.inference_mode = 'SingleImage'
            elif input[-4:] in ['.mp4', '.mkv', '.avi']:
                self.input = input
                self.inference_mode = 'Video'
            else:
                print("Invalid input file. The file should be an image or a video !!")
                exit()
        else:
            print("Input file doesn't exist. Check the input path")
            exit()
                 
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
                exit()
        else:
            print("Model weights file does not exist. Check the weights path")
            exit()
        
        # Checking output
        if output == None:
            self.output = self.input.split('/')[-1]
        else:
            self.output = output
        
        self.img_size = imgSize
        device = torch.device('cuda:0')
        cudnn.benchmark = True
        
        # Load models
        if self.inference_backend == 'PyTorch':
            self.model = torch.hub.load(
                'yolo_v5_main_files',
                'custom',
                path=self.model_weights,
                source='local'
            )
            self.model.eval()
            self.model.cuda()
            self.model.half()

        else:
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.INFO)
            with open(self.model_weights, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            self.bindings = dict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                #print(f"Shape: {shape}")
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            self.binding_addrs = {n: d.ptr for n, d in self.bindings.items()}
            self.context = model.create_execution_context()
            self.batch_size = self.bindings['images'].shape[0]
            #print(self.batch_size)

        # Initialize Tracker
        self.Objtracker = Sort(max_age=30, min_hits=7, iou_threshold=0.15)
        self.Objtracker.reset_count()


        # Running inference on different types of input
        if self.inference_mode == 'Video':
            self.VideoInference()
    
    def InferFrame(self):
        if self.inference_backend == 'PyTorch':
            results = self.model(self.frame, size=self.img_size)
            return results.xyxy[0]

    def VideoInference(self):
        video_capture = cv2.VideoCapture(self.input)

        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        codec = 'mp4v'

        video_output = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*codec), float(fps), (width, height),)

        Visualize = Visualizer()
        
        while video_capture.isOpened():
            _, self.frame = video_capture.read()

            if _:
                output = self.InferFrame()
                video_output.write(Visualize.drawBBOX(output, self.frame))
            else:
                break
        
        video_capture.release()
        video_output.release()

    
    

Inference(
    '/media/mydisk/videos/input_new/29/29.mp4', 
    '/home/students-fkk/Traffic_Camera_Tracking/tl_l6_89k_bs24_im1408_e150.pt',
    '/media/mydisk/videos/output_e150/29.mp4'
)