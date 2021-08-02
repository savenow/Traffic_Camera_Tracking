import torch, torchvision
import detectron2

# import some common libraries
import numpy as np
import os, json, cv2, random
from os import path

# import some common detectron2 utilitie s
from detectron2 import model_zoo
from detectron2.config import get_cfg, LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

import json


class setup_cfg():
    def __init__(self, pre_config_path, weights_path, test_dataset_path, cfg_mode='.py'):
        self.weights_path = weights_path
        self.pre_config_path = pre_config_path
        self.test_dataset = test_dataset_path

        self.metadata = MetadataCatalog.get('escooter_test')
        self.metadata.set(
            thing_classes=["Escooter"],
            thing_dataset_id_to_contiguous_id={1: 0},
            thing_colors=[(255, 99, 99)]    
        )

        if cfg_mode == '.yaml':
            self.cfg = self.setup_cfg()
        elif cfg_mode == '.py':
            self.cfg = self.setup_cfg_py()
        else:
            raise RuntimeError("Invalid cfg config: Only available configs are 'py' and 'yaml' !!!")


    
    def setup_cfg_yaml(self):
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
        
        argument_list = [
            # '--config-file', pre_config_path, 
            #'dataloader.train.dataset.names="escooter_train"',
            'dataloader.test.dataset.names="escooter_test"', 
            'dataloader.train.total_batch_size=1',
            f'train.init_checkpoint={self.weights_path}',
            f'train.output_dir="{self.output_path}"',
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
