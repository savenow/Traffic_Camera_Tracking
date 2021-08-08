import torch, torchvision
from IPython.display import Image, display

import detectron2

import logging
from detectron2.utils.logger import setup_logger
setup_logger()
logger = logging.getLogger("detectron2")

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

#from setup_cfg import setup_cfg
from OCR import extract_date_time

import json
from copy import deepcopy
import random
from time import sleep
import pytesseract

# Link: https://www.programmersought.com/article/76453652970/

class MyVisualizer(Visualizer):
    def _jitter(self, color):
        return color


class Main():

    def __init__(self, pre_config_path, weights_path, test_dataset_path, video_folder_path, output=None, cfg='.yaml', scale=0.7):

        self.input_files_path = video_folder_path
        
        # Currently this variable is redundant
        self.output_path = output
        self.scale = scale
        
        self.weights_path = weights_path
        self.pre_config_path = pre_config_path
        self.test_dataset = test_dataset_path

        cfg = self.loadcfg()
        #cfg = self.setup_cfg_py()
        self.predictor = DefaultPredictor(cfg)
        
        self.__process()

    def loadcfg(self):
        self.metadata = MetadataCatalog.get('escooter_test')
        self.metadata.set(
            thing_classes=["Escooter"],
            thing_dataset_id_to_contiguous_id={1: 0},
            thing_colors=[(255, 99, 99)]    
        )

        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.pre_config_path))
        
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

        return cfg

    def setup_cfg_py(self):
        self.metadata = MetadataCatalog.get('escooter_test')
        self.metadata.set(
            thing_classes=["Escooter"],
            thing_dataset_id_to_contiguous_id={1: 0},
            thing_colors=[(255, 99, 99)]    
        )

        # To prevent repeated registering of the same dataset
        try:
            register_coco_instances('escooter_test', {}, self.test_dataset, '')
        except AssertionError:
            pass

        argument_list = [
            # '--config-file', pre_config_path, 
            #'dataloader.train.dataset.names="escooter_train"',
            'dataloader.test.dataset.names="escooter_test"', 
            'dataloader.train.total_batch_size=1',
            f'train.init_checkpoint={self.weights_path}',
            'train.max_iter=8000',
            #'dataloader.train.warmup_length=800',
            'dataloader.train.num_workers=1',
            'optimizer.lr=0.00025',
            #'dataloader.train.num_classes=1',
            'model.roi_heads.num_classes=1',
            "model.backbone.bottom_up.stages.norm='BN'",
            "model.backbone.bottom_up.stem.norm='BN'",
            "model.backbone.norm='BN'",
        ]

        cfg = LazyConfig.load(self.pre_config_path)
        cfg = LazyConfig.apply_overrides(cfg, argument_list)

        return instantiate(cfg)

    def OCR(self, frame):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Users\balaji\AppData\Local\Programs\Tesseract-OCR\tesseract'

        # date_time_coordinates: (0, 0) till (570, 40)
        text_roi = frame[0:40, 0:570]

        # convert to gray
        gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)

        # threshold the grayscale image - The values are found out by trial and error method
        ret, thresh = cv2.threshold(gray,237,255,0)

        date_time = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6')[4:-1]
        return date_time

    def __process(self):
        output_file_folder = self.input_files_path + '//infered_with_background//'
        if not os.path.exists(output_file_folder):
            os.makedirs(output_file_folder)
        txt_file_path = output_file_folder + 'instance.txt'

        for videos in os.listdir(self.input_files_path):  
            
            if videos[-4:] in ['.mkv', '.mp4', '.avi']:
            #if videos == '08-06-2021_13-40.mkv':
                
                video_path = self.input_files_path + f'//{videos}'
                output_video_path = output_file_folder + videos

                with open(txt_file_path, 'a') as txt:
                    text = 'Processing Video File: ' + self.input_files_path + output_video_path + '\n'
                    txt.write(text)  
                    

                if os.path.isfile(output_video_path):
                    print(f'Skipping the video file: {videos}')
                    continue
                video_capture = cv2.VideoCapture(video_path)
                
                width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * self.scale)
                height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * self.scale)
                fps = int(video_capture.get(cv2.CAP_PROP_FPS))
                codec = 'mp4v'

                video_output = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*codec), float(fps), (width, height),)

                # Two bool variables for timestamp
                isEscooterPresent_previous = False
                isEscooterPresent_current = False

                framecount = 0
                while video_capture.isOpened():
                    _, frame = video_capture.read()
                    if _:   
                        isEscooterPresent_previous = isEscooterPresent_current

                        framecount += 1
                        output = self.predictor(frame)
                        Visualize = MyVisualizer(
                            frame[:, :, ::-1],
                            metadata=self.metadata, 
                            scale=self.scale, 
                            instance_mode=ColorMode.SEGMENTATION
                        )
                        output_img = Visualize.draw_instance_predictions(output["instances"].to("cpu")).get_image()[:, :, ::-1]
                        video_output.write(output_img)
                        print(f'Writing Frame Successfull')

                        num_instances = len(output['instances'])
                        print(f'Frame: {framecount}, number of instances: {num_instances}')
                        
                        if num_instances >= 1:
                            isEscooterPresent_current = True
                            date_time = extract_date_time(frame)
                                        
                        # Simple Logic to to store the first timestamp and the last timestamp an instance appears in the frame
                            if not isEscooterPresent_previous and isEscooterPresent_current:
                                with open(txt_file_path, 'a') as txt:
                                    txt.write(date_time)
                                isEscooterPresent_current = True
                                print(f'Writing OCR Successfull - Intake')

                        else:
                            print('No Instances detected')
                            print(f'Previous: {isEscooterPresent_previous}, Current: {isEscooterPresent_current}')
                        
                            isEscooterPresent_current = False   
                            if isEscooterPresent_previous:
                                with open(txt_file_path, 'a') as txt:
                                    outputText = '- ' + date_time + '\n'
                                    txt.write(date_time)
                                    print(f'Writing OCR Successfull - Outtake')

                    else:
                        break
                
                video_capture.release()
                video_output.release()
                print(f'Successfully saved the video file: {videos}')
                #break


def main():
    model_weights = path.abspath(r"C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\2_7250\model_final.pth")
    video_path = path.abspath(r"C:\Vishal-Videos\Project_Escooter_Tracking\input\33\33.mp4")
    inference_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Infered_Videos\2_7250_3.mp4')
    #metadata_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\metadata.json')
    #config_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\config.yaml')
    test_dataset_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Test.json')
    pre_config = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    pre_config_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Detectron2_New\detectron2\configs\new_baselines\mask_rcnn_R_101_FPN_400ep_LSJ.py')
    #model_weights_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\new-baseline-400ep\new_baseline_R101_FPN_Base.pkl')
    model_weights_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\With_Background_Images\model_final.pth')
    output_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\new-baseline-400ep')
    
    video_samples_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\samples')
    video_sample_1 = video_samples_path + '\\08-06-2021_08-00.mkv'
    
    #inference = Inference_Test(model_weights_new, test_dataset_path, video_sample_1)

    inference = Main(pre_config, model_weights_new, test_dataset_path, video_samples_path, output=output_new, cfg='.yaml')
    #inference = Main(pre_config_new, model_weights_new, test_dataset_path, video_samples_path, output=output_new, cfg='.yaml')
    
    #inference.save(output_path=inference_path, scale=0.7)

main()