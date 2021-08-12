import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, LazyConfig, CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from omegaconf import OmegaConf

# import some common libraries
import numpy as np
import os, json, cv2, random
from os.path import abspath
import json
from copy import deepcopy
import random


def processAnnotations(input_path=r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new'):
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
        },
        {
            "id": 2,
            "name": "Pedestrian",
            "supercategory": ""
        },
        {
            "id": 3,
            "name": "Cyclist",
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
    #input_path = r'C:\Vishal-Videos\Project_Escooter_Tracking\input'
    main_file_path = input_path

    # Declaration of empty lists that is later appended it with images and annotations.
    images_list_with_Anno = []
    annotations_list = []

    # Each image and annotations has an ID associated with it and it starts with 1.
    # These values are incremented as the images and annotations are being added.
    img_num = 1
    anno_num = 1

    escooter_count = 0 
    pedestrian_count = 0
    cyclist_count = 0

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
    for clips in os.listdir(main_file_path):
        # Checking that only numbers are given as folder names for the clips
        if all(char.isdigit() for char in clips):
            # Path of the clips folder
            clips_path = main_file_path + '\\' + clips
            # Path of the annotation of the clips
            
            annotation_file = clips_path + f'\\COCO\\annotations\instances_default.json'
            
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
                    "category_id": annotations['category_id'],
                    'segmentation': [],
                    'area': annotations['area'],
                    'bbox': annotations['bbox'],
                    'iscrowd': annotations['iscrowd']
                }

                # In the COCO-Format, every images and associated annotations are passed as array of dicts.
                images_list_with_Anno.append(image_dict)
                annotations_list.append(anno_dict)

                #imgNumWithAnnotations.append(img_num)
                # Incrementing the Image ID and Annotation ID for each loop
                img_num += 1
                anno_num += 1

                if annotations['category_id'] == 1:
                    escooter_count += 1
                elif annotations['category_id'] == 2:
                    pedestrian_count += 1
                elif annotations['category_id'] == 3:
                    cyclist_count += 1

                
            file.close()
        
                    
            # !! Meant for testing purpose.
            # if clip_number == 1:
            #     break

    print(f'\n- Total no.of annotations in the dataset: {anno_num}')

    print(f'\n- Total no.of escooter instances: {escooter_count}')
    print(f'\n- Total no.of pedestrians instances: {pedestrian_count}')
    print(f'\n- Total no.of cyclist instances: {cyclist_count}')
    
    train_json = deepcopy(coco_format)
    valid_json = deepcopy(coco_format)
    test_json = deepcopy(coco_format)

    train_split = 0.85
    valid_split = 0.15
    #test_split = 0.1

    # Function to split the whole dataset of images and annotations into train,
    # valid and test sets
    def splitDataset(img_Anno, annotations, trainSplit):
        
        trainSize = int(len(img_Anno) * trainSplit)
        train_images = []
        train_annotations = []
        
        valid_images = list(img_Anno)
        valid_annotations = list(annotations)
        while len(train_images) < trainSize:            
            index = random.randrange(len(valid_images))           
            train_images.append(valid_images.pop(index))
            train_annotations.append(valid_annotations.pop(index))
        
        # copySize = int(len(copy_images) * (validSplit/(1 - trainSplit)))
        # valid_images = []
        # valid_annotations = []

        # test_images = copy_images
        # test_annotations = copy_annotations
        # while len(valid_images) < copySize:
        #     index = random.randrange(len(test_images))
        #     valid_images.append(test_images.pop(index))
        #     valid_annotations.append(test_annotations.pop(index))

        
        
        # return [(train_images, train_annotations), (valid_images, valid_annotations), (test_images, test_annotations)]
        return [(train_images, train_annotations), (valid_images, valid_annotations)]

    #train_set, valid_set, test_set = splitDataset(images_list_with_Anno, images_list_without_Anno, annotations_list, 0.8, 0.1)
    train_set, valid_set = splitDataset(images_list_with_Anno, annotations_list, trainSplit=train_split)
    print("\n- Splitting the dataset into Train, Valid and Test is successfull\n")

    # Storing the processed arrays of images and annotations with their
    # respective keys in the final dataset
    # coco_format["images"] = images_list
    # coco_format["annotations"] = annotations_list

    train_json['images'] = train_set[0]
    train_json['annotations'] = train_set[1]

    valid_json['images'] = valid_set[0]
    valid_json['annotations'] = valid_set[1]

    # test_json['images'] = test_set[0]
    # test_json['annotations'] = test_set[1]

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
    # test_file = f"{input_path}\\{base_filename[:-5]}_Test.json"

    print("- Saving train, test and valid annotation files")
    with open(train_file, "w") as file:
        json.dump(train_json, file)
        print(f"  - Final training set file saved as: {train_file}")

    with open(valid_file, "w") as file:
        json.dump(valid_json, file)
        print(f"  - Final valid set file saved as: {valid_file}")

    # with open(test_file, "w") as file:
    #     json.dump(test_json, file)
    #     print(f"  - Final test set file saved as: {test_file}")

    # loadDataset(train_file, valid_file, test_file)
    loadDataset(train_file, valid_file)
    

def loadDataset(train, valid):
    # Registers the annotation from the json files 
    try:
        register_coco_instances("escooter_train", {}, train, '')
    except AssertionError:
        pass

    try:
        register_coco_instances("escooter_valid", {}, valid, '')
    except AssertionError:
        pass


def visualize(output_path):
    
    dataset_dicts = DatasetCatalog.get("escooter_train")
    escooter_metadata = MetadataCatalog.get("escooter_train")

    # Making a test_labels folder
    # sample_label_path = f'{output_path}\sample_labels'
    # os.system(f'mkdir {sample_label_path}')

    # For Checking how data is stored in Detectron2
    # for d in dataset_dicts:
    #     filename = d['file_name']
    #     count = 0
    #     for f in dataset_dicts:
    #         if f['file_name'] == filename:
    #             count += 1
    #             if count >= 2:
    #                 print(d['file_name'])
    #                 print(d['annotations'])
    #                 print(f['annotations'])

    # exit()

    # Just selecting 8 random images for visualizing purposes. Reduce scale variable if taking long time to load.
    for d in random.sample(dataset_dicts, 8):   
        
        img = cv2.imread(d['file_name'])

        visualizer = Visualizer(img[:, :, ::-1], scale=1)
        out = visualizer.draw_dataset_dict(d)

        out_filename = sample_label_path + '\\' + d['file_name'][:-4].split('\\')[-1] + '.png'
        cv2.imwrite(out_filename, out.get_image()[:, :, ::-1])

def loadConfig(weights_output_path):
        
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    
    #cfg.merge_from_file(model_zoo.get_config("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"))
    # cfg = model_zoo.get_config_file('new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py')
    # cfg = LazyConfig.load(cfg)
    # overrides = [
    #     'dataloader.train.dataset.names="escooter_train"', 
    #     'dataloader.test.dataset.names="escooter_test"', 
        
    #     f'train.init_checkpoint={weights_output_path + "/new_baseline_R101_FPN_Base.pkl"}',
    #     f'train.output_dir="{weights_output_path}"',
    #     'train.max_iter=8000',
    #     'dataloader.train.warmup_length=800',
    #     'dataloader.train.num_workers=1',
    #     'optimizer.lr=0.00025',
    #     'dataloader.train.num_classes=1',
    #     'model.roi_heads.num_classes=1',
        
    # ]
    # LazyConfig.apply_overrides(cfg, overrides)
    # #LazyConfig.save(cfg, weights_output_path + '/base_config.txt')
    # print(cfg)
    # with open(weights_output_path + '/base_config.json', 'w') as f:
    #     json.dump(cfg, f)
     
    # #cfg.merge_from_file('https://github.com/facebookresearch/detectron2/blob/master/configs/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py')
    # #cfg.merge_from_file(model_zoo.get("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py"))
    # # model = model_zoo.get("new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ.py")
    # return

    cfg = get_cfg()
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("escooter_train",)
    cfg.DATASETS.TEST = ('escooter_valid',)
    cfg.DATALOADER.NUM_WORKERS = 0

    #cfg.MODEL.WEIGHTS = r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Pre_Training_Model_Weights\Mask_RCNN_R_101_400ep.pkl'
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Initial Model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Initial Model

    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.WARMUP_ITERS = 800
    cfg.SOLVER.MAX_ITER = 6000   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = (4000, 5500)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (escooter)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.OUTPUT_DIR = weights_output_path
    # Specific for adding background images
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    with open(weights_output_path + '\\config.yaml', 'w') as f:
     f.write(cfg.dump())
  
    return cfg

def train(weights_path, config_file):
    coco_path = weights_path + '\\coco-eval'
    os.makedirs(coco_path, exist_ok=True)
    
    class CocoTrainer(DefaultTrainer):
        
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                os.makedirs(coco_path, exist_ok=True)
                output_folder = coco_path
            return COCOEvaluator(dataset_name, cfg, False, output_folder)

    trainer = CocoTrainer(config_file)
    trainer.resume_or_load(resume=False)
    trainer.train()


def main():
    # test_path = abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Test.json')
    # train_path = abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Train.json')
    # valid_path = abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Valid.json')

    # test_path = abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Test.json')
    train_path = abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new\Test_1_Train.json')
    valid_path = abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new\Test_1_Valid.json')

    model_path = abspath(r"C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\New_Annotations")

    # We are skipping the processing annotation step as it has already been done
    loadDataset(train=train_path, valid=valid_path)
    
    # processAnnotations()
    # Visualize some examples from the dataset
    #visualize(r'C:\Vishal-Videos\Project_Escooter_Tracking\\')

    config_file = loadConfig(model_path)
    train(model_path, config_file)

main()
    