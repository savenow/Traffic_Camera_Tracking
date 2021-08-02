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
from detectron2.config import get_cfg, LazyConfig, instantiate
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Instances

from setup_cfg import setup_cfg
from OCR import OCR

import json
from copy import deepcopy
import random
from time import sleep


# Link: https://www.programmersought.com/article/76453652970/

class MyVisualizer(Visualizer):
    def _jitter(self, color):
        return color


class Main():

    def __init__(self, pre_config_path, weights_path, test_dataset_path, input, output=None, cfg='py'):

        self.input_path = input
        self.output_path = output

        self.cfg = setup_cfg(pre_config_path=pre_config_path, weights_path=weights_path, test_dataset_path=test_dataset_path)
        self.predictor = DefaultPredictor(self.cfg)
        


    def __process(self):
        video_capture = cv2.VideoCapture(self.input_path)
        while video_capture.isOpened():
            _, frame = video_capture.read()
            if _:   
                output = self.predictor(frame)
                if len(output['instances'] >= 1):
                    date_time = OCR(frame)
                    print date_time
            else:
                break


def main():
    model_weights = path.abspath(r"C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\2_7250\model_final.pth")
    video_path = path.abspath(r"C:\Vishal-Videos\Project_Escooter_Tracking\input\33\33.mp4")
    inference_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Infered_Videos\2_7250_3.mp4')
    #metadata_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\metadata.json')
    #config_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\config.yaml')
    test_dataset_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Test.json')
    pre_config = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    pre_config_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Detectron2_New\detectron2\configs\new_baselines\mask_rcnn_R_101_FPN_400ep_LSJ.py')
    model_weights_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\new-baseline-400ep\new_baseline_R101_FPN_Base.pkl')
    output_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\new-baseline-400ep')
    
    video_samples_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\samples')
    video_sample_1 = video_samples_path + '\\08-06-2021_08-00.mkv'
    
    inference = Inference(model_weights_new, test_dataset_path, video_sample_1, mode='Video')
    inference.show(scale=0.7)
    #inference.save(output_path=inference_path, scale=0.7)

#main()