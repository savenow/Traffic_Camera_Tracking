#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.
This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.
Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg, LazyConfig, CfgNode, instantiate
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils import comm

logger = logging.getLogger("detectron2")

# import some common libraries
import numpy as np
import os, json, cv2, random
from os.path import abspath
import json
from copy import deepcopy
import random

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        optimizer=optim,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main_train(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        print(cfg)
        return
        do_train(args, cfg)


def loadDataset(train, valid, test):
    # Registers the annotation from the json files 
    try:
        register_coco_instances("escooter_train", {}, train, '')
    except AssertionError:
        pass

    try:
        register_coco_instances("escooter_valid", {}, valid, '')
    except AssertionError:
        pass
        
    try:
        register_coco_instances("escooter_test", {}, test, '')
    except AssertionError:
        pass

if __name__ == '__main__':
    test_path = abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Test.json')
    train_path = abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Train.json')
    valid_path = abspath(r'C:\Vishal-Videos\Project_Escooter_Tracking\input\Test_1_Valid.json')

    model_path = abspath(r"C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\Notebooks\Model_Weights_Loop_Test\new-baseline-400ep")

    # We are skipping the processing annotation step as it has already been done
    loadDataset(train=train_path, test=test_path, valid=valid_path)

    argument_list = [
        '--config-file', r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Detectron2_New\detectron2\configs\new_baselines\mask_rcnn_R_101_FPN_400ep_LSJ.py', 
        'dataloader.train.dataset.names="escooter_train"',
        'dataloader.test.dataset.names="escooter_test"', 
        
        f'train.init_checkpoint={model_path + "/new_baseline_R101_FPN_Base.pkl"}',
        f'train.output_dir="{model_path}"',
        'train.max_iter=8000',
        #'dataloader.train.warmup_length=800',
        'dataloader.train.num_workers=1',
        'optimizer.lr=0.00025',
        #'dataloader.train.num_classes=1',
        'model.roi_heads.num_classes=1',
    ]
    args = default_argument_parser().parse_args(argument_list)
    
    
    launch(
        main_train,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
