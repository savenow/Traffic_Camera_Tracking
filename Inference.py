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

def loadDataset():
  # !! Note !! -> Copied directly from the Jupyter Notebook and hence may contain some redundant code or unnecessary code which later can be deleted


  # coco_format is the dict file which includes all the values that needs to be output in the final annotations json file
  # Some of the key values like 'licenses', 'info' and 'categories' are constant and declared at first here

  coco_format = {
      "licenses": [{
          "name": "",
          "id": 0,
          "url": ""
      }],
      "info": {
          "contributor": "Vishal Balaji",
          "date_created": "",
          "description": "Escooter Dataset",
          "url": "",
          "version": "",
          "year": ""
      },
      "categories": [{
          "id": 1,
          "name": "Escooter",
          "supercategory": ""
      }]
  }

  # The key values 'images' and 'annotations' needs to be processed and appended. The below given lines is the format for
  # those dicts.
  """
  "images":[
      {
          "id":1,
          "width": 1920,
          "height": 1080,
          "file_name":"sdfa.PNG",
          "license":0,
          "flickr_url": "",
          "coco_url": "",
          "date_captured": 0
      }
  ]

  "annotations":[
      {
          "id": 1,
          "image_id": 55,
          "category_id": 1,
          "segmentation": [[]],
          "area": {some area number in float},
          "bbox": [].
          "iscrowd": 0
      }
  ]
  """

  # Path where the annotations are stored, when the repo is the path of current working directory
  #main_file_path = os.path.abspath(r'D:\Carissma Video Copy\Traffic Camera Tracking\Finished')
  input_path = r'C:\Vishal-Videos\Project_Escooter_Tracking\input'
  main_file_path = input_path

  # Declaration of empty lists that is later appended it with images and annotations.
  images_list = []
  annotations_list = []

  # Each image and annotations has an ID associated with it and it starts with 1.
  # These values are incremented as the images and annotations are being added.
  img_num = 1
  anno_num = 1

  def adjustFrameDifference(file_name, offset=1):
      # Adjusting for difference in frame
      file_name_from_dict = file_name.split('.')[0]
      file_number = int(file_name_from_dict[-6:])
      
      # 1 is the offset number for the frame difference between the annotations 
      # from CVAT and frames extracted from the FFMPEG Script
      file_number += 1
      
      # Adding the write number of 0's and taking care of proper filename
      if int(file_number / 10) == 0:
        new_file_name = file_name_from_dict[:-6] + '00000' + str(file_number) + '.png'
      elif int(file_number / 100) == 0:
        new_file_name = file_name_from_dict[:-6] + '0000' + str(file_number) + '.png'
      elif int(file_number / 1000) == 0:
        new_file_name = file_name_from_dict[:-6] + '000' + str(file_number) + '.png'
      elif int(file_number / 10000) == 0:
        new_file_name = file_name_from_dict[:-6] + '00' + str(file_number) + '.png'
      
      return new_file_name


  print("- Processing the following annotation files: ")
  for clip_number, clips in enumerate(os.listdir(main_file_path)):
      # Checking that only numbers are given as folder names for the clips
      if all(char.isdigit() for char in clips):
        # Path of the clips folder
        clips_path = main_file_path + '\\' + clips
        # Path of the annotation of the clips
        annotation_file = clips_path + f'\\{str(clips)}_Annotations.json'

        file = open(annotation_file)
        json_file = json.load(file)
        print(f'  - {annotation_file}')
          
        
        # !! Testing purpose only for restricting number of annotations
        # flag = 1
        for annotations in json_file['annotations']:

            anno_image_ID = annotations['image_id']
            anno_ID = annotations['id']

            image_filename = ''
            for images in json_file['images']:
                if images['id'] == anno_image_ID:
                    image_filename = images['file_name']

            filename = input_path + '\\' + clips + '\\images\\' + image_filename
            filename = adjustFrameDifference(filename)  
            
          # The formats for 'images' dictionary and 'annotations' dictionary in COCO
            image_dict = {
                'id': img_num,
                "width": 1920,
                "height": 1080,
                "file_name": filename,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0
            }
            anno_dict = {
                "id": anno_num,
                'image_id': img_num,
                "category_id": 1,
                'segmentation': annotations['segmentation'],
                'area': annotations['area'],
                'bbox': annotations['bbox'],
                'iscrowd': annotations['iscrowd']
            }

            # In the COCO-Format, every images and associated annotations are passed as array of dicts.
            images_list.append(image_dict)
            annotations_list.append(anno_dict)

            # Incrementing the Image ID and Annotation ID for each loop
            img_num += 1
            anno_num += 1
          
        file.close()

        # !! Meant for testing purpose.
        # if clip_number == 1:
        #     break

  print(f'\n- Total no.of annotations/images in the dataset: {anno_num}')

  train_json = deepcopy(coco_format)
  valid_json = deepcopy(coco_format)
  test_json = deepcopy(coco_format)

  train_split = 0.8
  valid_split = 0.1
  test_split = 0.1

  # Function to split the whole dataset of images and annotations into train,
  # valid and test sets
  def splitDataset(images, annotations, trainSplit, validSplit):
    trainSize = int(len(images) * trainSplit)
    train_images = []
    train_annotations = []
    
    copy_images = list(images)
    copy_annotations = list(annotations)
    while len(train_images) < trainSize:
      index = random.randrange(len(copy_images))
      train_images.append(copy_images.pop(index))
      train_annotations.append(copy_annotations.pop(index))
    

    copySize = int(len(copy_images) * (validSplit/(1 - trainSplit)))
    valid_images = []
    valid_annotations = []

    test_images = copy_images
    test_annotations = copy_annotations
    while len(valid_images) < copySize:
      index = random.randrange(len(test_images))
      valid_images.append(test_images.pop(index))
      valid_annotations.append(test_annotations.pop(index))
    
    return [(train_images, train_annotations), (valid_images, valid_annotations), (test_images, test_annotations)]

  train_set, valid_set, test_set = splitDataset(images_list, annotations_list, 0.8, 0.1)
  print("\n- Splitting the dataset into Train, Valid and Test is successfull\n")

  # Storing the processed arrays of images and annotations with their
  # respective keys in the final dataset
  # coco_format["images"] = images_list
  # coco_format["annotations"] = annotations_list

  train_json['images'] = train_set[0]
  train_json['annotations'] = train_set[1]

  valid_json['images'] = valid_set[0]
  valid_json['annotations'] = valid_set[1]

  test_json['images'] = test_set[0]
  test_json['annotations'] = test_set[1]

  # Code Snippet to automatically create new names for the many
  # .json files created during the testing
  base_filename = 'Test_'
  for numbers in range(20):
      check_filename = base_filename + str(numbers+1) + '.json'
      if check_filename not in os.listdir(os.getcwd()):
          base_filename = check_filename
          break


  # These lines writes all the dictionaries into the final required .json file
  # For train, valid and test individually
  train_file = f"{input_path}\\{base_filename[:-5]}_Train.json"
  valid_file = f"{input_path}\\{base_filename[:-5]}_Valid.json"
  test_file = f"{input_path}\\{base_filename[:-5]}_Test.json"


  print("- Saving train, test and valid annotation files")
  with open(train_file, "w") as file:
      json.dump(train_json, file)
      print(f"  - Final training set file saved as: {train_file}")

  with open(valid_file, "w") as file:
      json.dump(valid_json, file)
      print(f"  - Final valid set file saved as: {valid_file}")

  with open(test_file, "w") as file:
      json.dump(test_json, file)
      print(f"  - Final test set file saved as: {test_file}")

  
  register_coco_instances("escooter_train", {}, train_file, '')
  register_coco_instances("escooter_valid", {}, valid_file, '')
  register_coco_instances("escooter_test", {}, test_file, '')

  print('Registered all the datasets')



def setup_cfg(weights_path, config_file, test_dataset):
  # load config from file and command-line arguments
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
  cfg.merge_from_file(config_file)
  
  # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
  # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
  # add_panoptic_deeplab_config(cfg)
  # cfg.merge_from_file(args.config_file)
  # cfg.merge_from_list(args.opts)
  # Set score_threshold for builtin models
  # with open(test_dataset) as f:
  #   escooter_test_dict = json.load(f)

  
  cfg.DATASETS.TRAIN = ('escooter_train',)
  cfg.DATASETS.TEST = ('escooter_test',)
  # load weights
  cfg.MODEL.WEIGHTS = weights_path
  #cfg.MODEL.WEIGHTS = r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Model_metrics\Output_1\model_final.pth'
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
  cfg.freeze()
  with open('config_test.yaml', 'w') as f:
    f.write(cfg.dump())
  return cfg


def Visualize(cfg, input_path, output_path, meta_path):
  predictor = DefaultPredictor(cfg)

  video_capture = cv2.VideoCapture(input_path)
  video_output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (1920, 1080))
  
  #MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['escooter']
  Escooter_Metadata = MetadataCatalog.get('escooter_test')
  Escooter_Metadata.set(thing_classes=["Escooter"],
                        thing_dataset_id_to_contiguous_id={1: 0},    
  )
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
      print(outputs)
      #sleep(2)
      
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

  loadDataset()
  inference_cfg = setup_cfg(model_weights, config_path, test_dataset_path)
  Visualize(inference_cfg, video_path, inference_path, metadata_path)

main()