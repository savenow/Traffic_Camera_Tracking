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
import json
from copy import deepcopy
import random
from time import sleep
import pytesseract
from SORT_Tracking.sort import *

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

# Meant for SORT Tracker BBOX
color_boxes = [(255, 99, 99), (255, 120, 120), (255, 150, 150), (255, 80, 80)]

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
        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.pre_config_path))
        
        # To prevent repeated registering of the same dataset
        try:
            register_coco_instances('escooter_valid', {}, self.test_dataset, '')
        except AssertionError:
            pass

        self.metadata = MetadataCatalog.get('escooter_valid')
        self.metadata.set(
            thing_classes=["Escooter", "Pedestrian", "Cyclists"],
            thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2},
            thing_colors=[(255, 99, 99), (99, 255, 99), (99, 99, 255)]    
        )

        cfg.DATASETS.TEST = ('escooter_valid',)
        # cfg.DATALOADER.NUM_WORKERS = 0
        # # load weights
        # cfg.SOLVER.IMS_PER_BATCH = 1
        # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        # cfg.SOLVER.MAX_ITER = 9999 #3750   
        # cfg.SOLVER.WARMUP_ITERS = int(cfg.SOLVER.MAX_ITER/5)
        # cfg.SOLVER.STEPS = []        
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.87 
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
        cfg.freeze()

        return cfg
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

    def __preprocess_sort(self, input):
        detection = []
        for (bbox, score) in input:
            bbox = bbox.tolist()
            score = score.tolist()
            #print(f' SORT BBOX: {bbox}, score: {score}')
            bbox.append(score)
            detection.append(bbox)
            #print(f'Detection: {detection}')
        detection = np.array(detection)
        return detection
    
    def show_tracker_bbox_score(self, input, frame):
        img = frame
        for (detection, score) in input:
            # For bounding box
            print(f"Detection: {detection}, Score: {score}")
            tracker_id = int(detection[4])
            x1 = int(detection[0])
            y1 = int(detection[1])
            x2 = int(detection[2])
            y2 = int(detection[3])

            print(f"Tracker ID: {tracker_id}")
            print(f"X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}")

            color = (100, 100, 255)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            
            # For the text background
            # Finds space required by the text so that we can put a background with that amount of width.
            label = f'Track_ID: {tracker_id}, {int(score * 100)}%'
            (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            # Prints the text.    
            img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            text_color = (0, 0, 0)
            img = cv2.putText(img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, lineType=cv2.LINE_AA)
            
        return img

    def __process(self):
        #output_file_folder = self.input_files_path + '//infered_with_background//'
        output_file_folder = self.output_path
        if not os.path.exists(output_file_folder):
            os.makedirs(output_file_folder)
        txt_file_path = output_file_folder + '/instance.txt'

        for videos in os.listdir(self.input_files_path):  
            
            if videos[-4:] in ['.mkv', '.mp4', '.avi']:
            #if videos == '8_00.mp4':
                
                video_path = self.input_files_path + f'//{videos}'
                output_video_path = output_file_folder +'/' + videos

                with open(txt_file_path, 'a') as txt:
                    text = 'Processing Video File: ' + self.input_files_path + output_video_path + '\n'
                    txt.write(text)  
                    

                if os.path.isfile(output_video_path):
                    print(output_video_path)
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

                # Sort Tracker
                Object_tracker = Sort()

                framecount = 0
                
                # Used for finding how many continous frames the instances are detected
                # In this script, this threshold is three for initiating SORT Algorithm
                num_frames_present = 0
                while video_capture.isOpened():
                    _, frame = video_capture.read()
                    if _:   
                        isEscooterPresent_previous = isEscooterPresent_current

                        framecount += 1
                        output = self.predictor(frame)
                    
                        #video_output.write(output_img)
                        #print(f'Writing Frame Successfull')

                        num_instances = len(output['instances'])
                        print(f'\nFrame: {framecount}, number of instances: {num_instances}')
                        
                        boxes = output['instances'].pred_boxes.tensor.cpu().numpy() if output['instances'].has("pred_boxes") else None
                        scores = output['instances'].scores.cpu().detach().numpy() if output['instances'].has("scores") else None
                        #print(f'BBOXES: \n{boxes}\nSCORES: \n{scores}')
                        
                        if num_instances >= 1:
                            num_frames_present += 1
                            if num_frames_present >= 3:
                                num_frames_present = 3
                            
                            if num_frames_present > 0: 
                                detections = self.__preprocess_sort(zip(boxes, scores))
                                track_bbs_ids = Object_tracker.update(detections)
                                
                                # cv2.imshow('Image', self.show_tracker_bbox_score(zip(track_bbs_ids, scores), output_img))
                                # cv2.waitKey(0)
                                output_tracker = self.show_tracker_bbox_score(zip(track_bbs_ids, scores), frame)
                                Visualize = MyVisualizer(
                                    output_tracker[:, :, ::-1],
                                    metadata=self.metadata, 
                                    scale=self.scale, 
                                    instance_mode=ColorMode.SEGMENTATION,
                                )
                                output_img = Visualize.draw_instance_predictions(output["instances"].to("cpu")).get_image()[:, :, ::-1]
                                video_output.write(output_img)
                            else:
                                Visualize = MyVisualizer(
                                    frame[:, :, ::-1],
                                    metadata=self.metadata, 
                                    scale=self.scale, 
                                    instance_mode=ColorMode.SEGMENTATION
                                )
                                output_img = Visualize.draw_instance_predictions(output["instances"].to("cpu")).get_image()[:, :, ::-1]
                                video_output.write(output_img)

                            isEscooterPresent_current = True
                            date_time = extract_date_time(frame)      
                            
                            # Simple Logic to to store the first timestamp and the last timestamp an instance appears in the frame
                            if not isEscooterPresent_previous and isEscooterPresent_current:
                                with open(txt_file_path, 'a') as txt:
                                    txt.write(date_time)
                                isEscooterPresent_current = True
                                print(f'Writing OCR Successfull - Intake')
                            #break
                        else:
                            num_frames_present = 0
                            track_bbs_ids = Object_tracker.update()
                            Visualize = MyVisualizer(
                                frame[:, :, ::-1],
                                metadata=self.metadata, 
                                scale=self.scale, 
                                instance_mode=ColorMode.SEGMENTATION
                            )
                            output_img = Visualize.draw_instance_predictions(output["instances"].to("cpu")).get_image()[:, :, ::-1]
                            video_output.write(frame)
                            print('No Instances detected')
                            
                            # Logging for timestamps
                            #print(f'Previous: {isEscooterPresent_previous}, Current: {isEscooterPresent_current}')
                        
                            isEscooterPresent_current = False   
                            if isEscooterPresent_previous:
                                with open(txt_file_path, 'a') as txt:
                                    outputText = '- ' + date_time + '\n'
                                    txt.write(outputText)
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
    test_dataset_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new\Test_1_Valid.json')
    pre_config = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    pre_config_new = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    #model_weights_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\new-baseline-400ep\new_baseline_R101_FPN_Base.pkl')
    model_weights_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\New_Annotations\model_final.pth')
    output_new = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\New_Annotations\inference_output\\')
    
    #video_samples_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\samples\re-encode')
    video_samples_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new\43')

    video_sample_1 = video_samples_path + '\\08-06-2021_08-00.mkv'
    
    #inference = Inference_Test(model_weights_new, test_dataset_path, video_sample_1)

    inference = Main(pre_config_new, model_weights_new, test_dataset_path, video_samples_path, output=output_new, cfg='.yaml')
    #inference = Main(pre_config_new, model_weights_new, test_dataset_path, video_samples_path, output=output_new, cfg='.yaml')
    
    #inference.save(output_path=inference_path, scale=0.7)

main()