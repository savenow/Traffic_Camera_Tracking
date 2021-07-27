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

from detectron2 import model_zoo

def setup_cfg(weights_path):
  # load config from file and command-line arguments
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
  # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
  # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
  # add_panoptic_deeplab_config(cfg)
  # cfg.merge_from_file(args.config_file)
  # cfg.merge_from_list(args.opts)
  # Set score_threshold for builtin models
  # load weights
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weights_path)
  #cfg.MODEL.WEIGHTS = r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Model_metrics\Output_1\model_final.pth'
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
  cfg.freeze()
  return cfg


def Visualize(cfg, input_path, output_path):
  predictor = DefaultPredictor(cfg)

  video_capture = cv2.VideoCapture(input_path)
  video_output = cv2.VideoOutput(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30)
  Escooter_Metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

  while video_capture.isOpened():
    _, frame = video_capture.read()
    if _:
      im = frame
      outputs = predictor(im)
      v = Visualizer(im[:, :, ::-1],
                      metadata=Escooter_Metadata, 
                      scale=0.8, 
                      instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
      )
      v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
      video_output.write(v.get_image()[:, :, ::-1])

  video_capture.release()
  video_output.release()

def main():
  model_weights = path(r"C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Model_metrics\Output_3_R101_FPN")
  video_path = path(r"C:\Vishal-Videos\Project_Escooter_Tracking\input\24\24.mp4")
  inference_path = path(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Infered_Videos\1.avi')

  inference_cfg = setup_cfg(model_weights)
  Visualize(inference_cfg, video_path, inference_path)








"""


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("escooter_train",)
cfg.DATASETS.TEST = ('escooter_test',)
cfg.DATALOADER.NUM_WORKERS = 0

cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.WARMUP_ITERS = 800
cfg.SOLVER.MAX_ITER = 7000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


# load weights
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#cfg.MODEL.WEIGHTS = r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Model_metrics\Output_1\model_final.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# Set training data-set path
cfg.DATASETS.TEST = ("escooter_valid", )
# Create predictor (model for inference)
predictor = DefaultPredictor(cfg)
dataset_dicts = DatasetCatalog.get("escooter_test")
escooter_metadata = MetadataCatalog.get("escooter_test")

valid_images = []
for d in random.sample(dataset_dicts, 6):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=escooter_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
   
    validation_label_path = f'{input_path}\\valid_labels'
    os.system(f'mkdir {validation_label_path}')

    out_filename = validation_label_path + '\\' + d['file_name'][:-4].split('\\')[-1] + '.png'
    cv2.imwrite(out_filename, v.get_image()[:, :, ::-1])
    valid_images.append(out_filename)

# Displays images in notebook. Caution: 'display()' may not work outside this notebook
for images in valid_images:
  display(Image(images))
"""