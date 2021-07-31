import torch, torchvision
from IPython.display import Image, display

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from os import path

# import some common detectron2 utilitie s
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances

import json
from copy import deepcopy
import random
from time import sleep

from detectron2 import model_zoo

# Link: https://www.programmersought.com/article/76453652970/

class MyVisualizer(Visualizer):
    def _jitter(self, color):
        return color


class Inference():

    def __init__(self, weights_path, test_dataset_path, input, output=None, mode='Image'):
        self.weights_path = weights_path
        self.test_dataset = test_dataset_path
        self.input_path = input
        self.output_path = output
        
        
        if mode == 'Image':
            self.isImage = True
        elif mode == 'Video':
            self.isImage = False
        else:
            raise RuntimeError("Invalid Mode: Only available modes are 'Video' and 'Image' !!!")

        self.cfg, self.metadata = self.setup_cfg()
        self.predictor = DefaultPredictor(self.cfg)
        

    def setup_cfg(self):
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        
        # To prevent repeated registering of the same dataset
        try:
            register_coco_instances('escooter_test', {}, self.test_dataset, '')
        except AssertionError:
            pass

        cfg.DATASETS.TEST = ('escooter_test',)
        cfg.DATALOADER.NUM_WORKERS = 0
        # load weights
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 3750   
        cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER/5)
        cfg.SOLVER.STEPS = []        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.87 
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
        cfg.freeze()

        Escooter_Metadata = MetadataCatalog.get('escooter_test')
        Escooter_Metadata.set(
            thing_classes=["Escooter"],
            thing_dataset_id_to_contiguous_id={1: 0},
            thing_colors=[(255, 99, 99)]    
        )

        return cfg, Escooter_Metadata


    def display(self, frame):
        cv2.imshow(self.Window_Name, frame)

        if self.isImage:
            cv2.waitKey(0)
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True
            else:
                return False

    def __process(self, scale, mode='display'):
        if mode == 'display':
            self.Window_Name = 'Output'
            cv2.namedWindow(self.Window_Name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.Window_Name, 1280, 720)
        
        if self.isImage:
            frame = cv2.imread(self.input_path)
            output = self.predictor(frame)
            if mode == 'output':
                return output
            Visualize = MyVisualizer(
                frame[:, :, ::-1],
                metadata=self.metadata, 
                scale=scale, 
                instance_mode=ColorMode.SEGMENTATION
            )      
            output_img = Visualize.draw_instance_predictions(output["instances"].to("cpu")).get_image()[:, :, ::-1]

            if mode == 'display':
                self.display(output_img)
            elif mode == 'save_clip':
                cv2.imwrite(self.output_path, output_img)
        
        elif self.isImage is False:
            video_capture = cv2.VideoCapture(self.input_path)

            if mode == 'save_clip':
                print('Saving Clip......')
                width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
                height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
                fps = int(video_capture.get(cv2.CAP_PROP_FPS))
                codec = 'mp4v'

                video_output = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*codec), float(fps), (width, height),)

            while video_capture.isOpened():
                _, frame = video_capture.read()
                if _:   
                    output = self.predictor(frame)
                    if mode == 'output':
                        if len(output['instances']) >= 1:
                            pass
                    Visualize = MyVisualizer(
                        frame[:, :, ::-1],
                        metadata=self.metadata, 
                        scale=scale, 
                        instance_mode=ColorMode.SEGMENTATION
                    )
                    
                    output_img = Visualize.draw_instance_predictions(output["instances"].to("cpu")).get_image()[:, :, ::-1]

                    if mode == 'display':
                        toClose = self.display(output_img)
                        if toClose:
                            break
                    elif mode == 'save_clip':
                        video_output.write(output_img)
                else:
                    break

            
            if mode == 'save_clip':     
                video_output.release()
            
            video_capture.release()

    def show(self, scale):
        self.__process(scale)

    def save(self, output_path, scale):
        self.output_path = output_path
        self.__process(scale, mode='save_clip')

    def output(self, scale):
        return self.__process(scale, mode='output')

def main():
    model_weights = path.abspath(r"C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\2_7250\model_final.pth")
    video_path = path.abspath(r"C:\Vishal-Videos\Project_Escooter_Tracking\input\33\33.mp4")
    inference_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Infered_Videos\2_7250_3.mp4')
    #metadata_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\metadata.json')
    #config_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\config.yaml')
    test_dataset_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Test.json')

    video_samples_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\samples')
    video_sample_1 = video_samples_path + '\\08-06-2021_08-00.mkv'
    
    inference = Inference(model_weights, test_dataset_path, video_sample_1, mode='Video')
    inference.show(scale=0.7)
    #inference.save(output_path=inference_path, scale=0.7)

#main()