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

def setup_cfg(weights_path, config_file, test_dataset):
  # load config from file and command-line arguments
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
  #cfg.merge_from_file(config_file)
  
  register_coco_instances('escooter_test', {}, test_dataset, '')
  cfg.DATASETS.TEST = ('escooter_test',)
  cfg.DATALOADER.NUM_WORKERS = 0
  # load weights
  cfg.SOLVER.IMS_PER_BATCH = 2
  cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
  cfg.SOLVER.WARMUP_ITERS = int(3750/5)
  cfg.SOLVER.MAX_ITER = 3750   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
  cfg.SOLVER.STEPS = []        # do not decay learning rate
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
  cfg.MODEL.WEIGHTS = weights_path
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # set the testing threshold for this model
  cfg.freeze()

  # with open('config_test.yaml', 'w') as f:
  #   f.write(cfg.dump())
  return cfg


def Visualize(cfg, input_path, output_path, meta_path):
  predictor = DefaultPredictor(cfg)

  video_capture = cv2.VideoCapture(input_path)
  width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(video_capture.get(cv2.CAP_PROP_FPS))
  codec = 'mp4v'

  video_output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), float(fps), (width, height),)
  
  #MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['escooter']
  Escooter_Metadata = MetadataCatalog.get('escooter_test')
  Escooter_Metadata.set(
    thing_classes=["Escooter"],
    thing_dataset_id_to_contiguous_id={1: 0},
    thing_colors=[(255, 0, 0)]    
  )
  print(Escooter_Metadata)

  framecount = 0
  while video_capture.isOpened():
    _, frame = video_capture.read()
    if _:
      outputs = predictor(frame)
      v = MyVisualizer(frame[:, :, ::-1],
                      metadata=Escooter_Metadata, 
                      scale=1, 
                      instance_mode=ColorMode.SEGMENTATION
      )
      
      print(len(outputs['instances']))
      print(outputs)
      #print(outputs['instances'].get_fields())
      
      v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      #video_output.write(v.get_image()[:, :, ::-1])

      framecount += 1
      # if framecount == 5:
      #   break
      print(framecount)

    else:
      break
    
  video_capture.release()
  video_output.release()

def main():
  model_weights = path.abspath(r"C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\2_7250\model_final.pth")
  video_path = path.abspath(r"C:\Vishal-Videos\Project_Escooter_Tracking\input\24\24.mp4")
  inference_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Infered_Videos\2_7250_3.mp4')
  metadata_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\metadata.json')
  config_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\config.yaml')
  test_dataset_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Test.json')

  inference_cfg = setup_cfg(model_weights, config_path, test_dataset_path)
  Visualize(inference_cfg, video_path, inference_path, metadata_path)


main()