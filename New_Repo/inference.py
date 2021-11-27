import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from sort_yoloV5 import Sort

class Inference():
    def __init__(self, input, output, model_weights):
        # self.input = input
        # self.output = output
        # self.model_weights = model_weights
        
        # Checking input
        if os.path.isfile(input):
            if input[-3:] in ['.png', '.jpg']:
               self.input = input
               self.inference_mode = 'SingleImage'
            elif input[-3:] in ['.mp4', '.mkv', '.avi']:
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
        
        device = torch.device('cuda:0')
        cudnn.benchmark = True
        
        if self.inference_backend == 'PyTorch':
            model = torch.hub.load(
                'yolo_v5_main_files',
                'custom',
                path=self.model_weights,
                source='local'
            )
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
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            self.binding_addrs = {n: d.ptr for n, d in self.bindings.items()}
            self.context = model.create_execution_context()
            self.batch_size = self.bindings['images'].shape[0]

