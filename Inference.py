import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

from detectron2.utils.visualizer import ColorMode
from IPython.display import Image, display

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from os import path

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

import json
from copy import deepcopy
import random
from time import sleep

from detectron2 import model_zoo


def setup_cfg(weights_path, config_file, test_dataset):
  # load config from file and command-line arguments
  cfg = get_cfg()
  cfg.merge_from_file(config_file)
  #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
  
  # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
  # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
  # add_panoptic_deeplab_config(cfg)
  # cfg.merge_from_file(args.config_file)
  # cfg.merge_from_list(args.opts)
  # Set score_threshold for builtin models
  with open(test_dataset) as f:
    escooter_test_dict = json.load(f)

  register_coco_instances("escooter_test", {}, test_dataset, '')
  
  cfg.DATASETS.TEST = ('escooter_test',)
  # load weights
  cfg.MODEL.WEIGHTS = weights_path
  #cfg.MODEL.WEIGHTS = r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Model_metrics\Output_1\model_final.pth'
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
  #cfg.freeze()
  return cfg


def Visualize(cfg, input_path, output_path, meta_path):
  predictor = DefaultPredictor(cfg)

  video_capture = cv2.VideoCapture(input_path)
  video_output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (1920, 1080))
  
  Escooter_Metadata = MetadataCatalog.get('escooter_test')
  print(Escooter_Metadata)

  framecount = 0
  while video_capture.isOpened():
    _, frame = video_capture.read()
    if _:
      outputs = predictor(frame)
      v = Visualizer(frame[:, :, ::-1],
                      metadata=Escooter_Metadata, 
                      scale=1, 
                      instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
      )
      #print(outputs)
      sleep(2)
      
      v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      #video_output.write(v.get_image()[:, :, ::-1])

      out_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Infered_Videos\labels_1')
      cv2.imwrite(f'{out_path}\\{framecount}.png', v.get_image()[:, :, ::-1])
      framecount += 1
      
      # if framecount % 20 == 0:
      #   cv2.imshow("Frame", v.get_image()[:, :, ::-1])
      #   cv2.waitKey(0)
      print(framecount)
    else:
      break
    
  video_capture.release()
  video_output.release()

def main():
  model_weights = path.abspath(r"C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\2_3750\model_final.pth")
  video_path = path.abspath(r"C:\Vishal-Videos\Project_Escooter_Tracking\input\24\24.mp4")
  inference_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Infered_Videos\1.avi')
  metadata_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\metadata.json')
  config_path = path.abspath(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\config.yaml')
  test_dataset_path = path.abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Test.json')

  inference_cfg = setup_cfg(model_weights, config_path, test_dataset_path)
  Visualize(inference_cfg, video_path, inference_path, metadata_path)

main()