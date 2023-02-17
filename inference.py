import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import math
from torch._C import device
from tqdm import tqdm
from collections import namedtuple, defaultdict
import torch.backends.cudnn as cudnn
import pandas as pd
from datetime import datetime, timedelta
from copy import deepcopy
import time
import yaml
import shutil

import sys
sys.path.append('./yolo_v5_main_files')
from models.common import DetectMultiBackend, AutoShape
from utils.datasets import LoadImages
from utils.torch_utils import time_sync
from utils.general import LOGGER, non_max_suppression, scale_coords, check_img_size, print_args
from hubconf import custom

from sort_yoloV5 import Sort
from visualizer import Visualizer, Minimap
from calibration import Calibration
from timestamp_ocr import OCR_TimeStamp
import traffic_light_region as light_state

# from extract_stored_detections_copy import PostProcess
from post_process_multiProcess import PostProcess

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class Inference():
    def __init__(self, input, model_weights, output, minimap, 
                trj_mode, disable_post_process, save_infer_video, 
                imgSize, update_rate, save_class_frames, traffic_light_info,light_region, night_mode):        
        # Main config
        main_config_path = 'configs/main_param.yaml'
        with open(main_config_path) as file_stream:
            try:
                self.main_config_dict = yaml.safe_load(file_stream)
            except yaml.YAMLError as exc:
                print(f'[Error] Failed to load main .yaml file. {exc}\n Quitting')
                exit()

        # Inference Params
        self.target_resolution = self.main_config_dict['target_resolution']
        self.img_size = self.main_config_dict['img_size']
        self.conf_thres = self.main_config_dict['conf_thres']
        self.iou_thres = self.main_config_dict['iou_thres']
        self.agnostic_nms = False
        self.max_det = self.main_config_dict['max_det']
        self.classes = None # Filter classes

        self.device = torch.device('cuda:0')
        self.half = True
        cudnn.benchmark = True
        
        self.update_rate = update_rate
        self.save_infer_video = save_infer_video
        self.showTrajectory = trj_mode
        self.trajectory_retain_duration = self.main_config_dict['trajectory_retain_duration'] # Number of frames the trajectory for each tracker id must be retained before removal
        self.save_class_frames = save_class_frames

        # Checking input
        if os.path.isfile(input):
            # Further functionality needs to be added for Folder Inference :))
            if input[-4:] in ['.png', '.jpg']:
               self.input = input
               self.inference_mode = 'SingleImage'
            elif input[-4:] in ['.mp4', '.mkv', '.avi']:
                self.input = input
                self.inference_mode = 'Video'
                self.fps = self.main_config_dict['fps']
            else:
                print("Invalid input file. The file should be an image or a video !!")
                exit(-1)
        else:
            print("Input file doesn't exist. Check the input path")
            exit(-1)
                 
        # Checking weights file
        if os.path.isfile(model_weights):
            if model_weights[-3:] == '.pt':
                self.model_weights = model_weights
                self.inference_backend = 'PyTorch'
            elif model_weights[-7:] == '.engine':
                self.model_weights = model_weights
                self.inference_backend = 'TensorRT'
            else:
                print(f"Invalid Weights file. {model_weights} does not end with '.engine' or '.pt'")
                exit(-1)
        else:
            print("Model weights file does not exist. Check the weights path")
            exit(-1)
        
        # Checking output
        if output == None:
            self.output = self.input.split('/')[-1]
        else:
            self.output = output
            __output_path_processing = Path(self.output)
            self.file_stem_name = __output_path_processing.stem
            self.parent_directory = __output_path_processing.parents[0]
            self.output_dir_path = self.parent_directory / self.file_stem_name
            if not os.path.exists(self.output_dir_path):
                os.makedirs(self.output_dir_path)
                os.makedirs(self.output_dir_path/"Save-frames")
            else:
                shutil.rmtree(self.output_dir_path)       # Removes all the subdirectories!
                os.makedirs(self.output_dir_path)
                os.makedirs(self.output_dir_path/"Save-frames")

        # Loading Model
        model = DetectMultiBackend(self.model_weights, device=self.device, dnn=None)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        self.names = ['Escooter', 'Pedestrian', 'Cyclist', 'Motorcycle', 'Car', 'Truck', 'Bus']
        
        if self.pt:
            model = model.model.half()
        self.model = model

        # Initialize Tracker
        self.Objtracker = Sort(max_age=self.main_config_dict['max_age'], min_hits=self.main_config_dict['min_hits'], iou_threshold=self.main_config_dict['iou_threshold'])
        self.Objtracker.reset_count()

        # Camera Calibration data: Used for velocity estimation
        self.enable_minimap = minimap
        self.enable_trajectory = trj_mode
        self.Calib = Calibration()
        
        # Parameters for velocity estimation
        self.trackDict = defaultdict(list)

        # For storing conversion values
        self.Minimap_storage = Minimap()

        # Intializing Traffic light region parameters
        self.traffic_light_info = traffic_light_info
        if self.traffic_light_info:
            self.light_region = light_region
            self.night_mode = night_mode
            self.state = light_state.ROI(self.input, self.light_region, self.night_mode, str(self.output_dir_path))

        # Main Inference
        self.runInference()
    
    def UpdateTracker(self, pred):
        if len(pred) > 0:
            dets = []
            for items in pred:
                dets.append(items[:].tolist())
        
            dets = np.array(dets)
            self.tracker = self.Objtracker.update(dets)
        else:
            self.tracker = self.Objtracker.update()
        
    def VelocityEstimation(self, velocity_array):    
        for detection in self.tracker:
            class_id = detection[5]
            center_x = (detection[0] + detection[2])/2 

            if class_id in (0,1,2):
                _, max_y = sorted((detection[1], detection[3]))
            elif class_id in (3,4,5,6):
                max_y = (detection[1] + detection[3])/2     # Center of bbox for classes other than Escooter, Cyclist, and Pedestrian

            trackID = int(detection[9])
            self.trackDict[trackID].append((int(center_x), int(max_y)))
            
            if len(self.trackDict[trackID]) > 10: 
                previous_point = self.Calib.projection_pixel_to_world(self.trackDict[trackID][-2])
                current_point = self.Calib.projection_pixel_to_world(self.trackDict[trackID][-1])
                previous_point = (previous_point[0],previous_point[1])
                current_point = (current_point[0],current_point[1])

                distance_metres = round(float(math.sqrt(math.pow(previous_point[0] - current_point[0], 2) + math.pow(previous_point[1] - current_point[1], 2))), 2)
                speed_kmH = round(float(distance_metres * self.fps * 3.6), 2)
                output_array = np.append(detection, [speed_kmH, self.trackDict[trackID][-2][0], self.trackDict[trackID][-2][1], self.trackDict])
                velocity_array.append(output_array)
                del self.trackDict[trackID][0]

        return velocity_array

    def UpdateStorage_withTracker(self, output_dictionary):
        output = []
        for detection in self.tracker:
            temp_dict = deepcopy(output_dictionary)

            temp_dict['Tracker_ID'] = int(detection[9])
            temp_dict['Class_ID'] = int(detection[5])
            temp_dict['Conf_Score'] = round(detection[4] * 100, 1)

            x1 = int(detection[0])
            y1 = int(detection[1])
            x2 = int(detection[2])
            y2 = int(detection[3])
            temp_dict['BBOX_TopLeft'] = (x1, y1)
            temp_dict['BBOX_BottomRight'] = (x2, y2)
            
            center_x = (x1+x2)/2        
            _, max_y = sorted((y1, y2))
            temp_dict['Minimap_Coordinates'] = self.Minimap_storage.projection_image_to_map_noScaling(center_x, max_y)
            output.append(temp_dict)
        return output
    
    def UpdateStorage_onlyYolo(self, output_dictionary, pred):
        output = []
        for detection in pred:
            temp_dict = deepcopy(output_dictionary)

            temp_dict['Tracker_ID'] = None
            temp_dict['Class_ID'] = int(detection[5].item())
            temp_dict['Conf_Score'] = round(detection[4].item() * 100, 1)
            
            x1 = int(detection[0])
            y1 = int(detection[1])
            x2 = int(detection[2])
            y2 = int(detection[3])
            temp_dict['BBOX_TopLeft'] = (x1, y1)
            temp_dict['BBOX_BottomRight'] = (x2, y2)

            center_x = (x1+x2)/2        
            _, max_y = sorted((y1, y2))
            temp_dict['Minimap_Coordinates'] = self.Minimap_storage.projection_image_to_map_noScaling(center_x, max_y)
            output.append(temp_dict)
        return output

    def runInference(self):
        dataset = LoadImages(self.input, img_size=self.img_size, stride=self.stride, auto=self.pt and not self.jit)
        bs = 1
        vid_path, vid_writer = None, None

        ocr = OCR_TimeStamp()
        ocr_vertical_offset = int((self.img_size[0]-self.target_resolution[0])/2) # Since the imgSize for inference is 1920x1920 and input video is 1920x1080, some padding is automatically applied by Yolo. Offsetting the y-values for this padding.
        output_data = [] # For writing detection/tracker data to .csv for post processing
        Visualize = Visualizer(self.enable_minimap, self.enable_trajectory, self.update_rate, self.trajectory_retain_duration, self.save_class_frames)
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        framecount = 0
        time_start = time_sync()
        for path, im, im0, vid_cap, s, videoTimer in dataset:
            framecount += 1 
            if framecount < -1:
                continue
            elif framecount > 108000:
                print('[Warning] Video sequence exceeds one hour. Stopping inference due to possible ram issues')
                break

            storing_output = {}
            storing_output["Video_Internal_Timer"] = videoTimer
            
            # OCR Reading Timestamp
            if self.main_config_dict['is_ocr_enabled']:
                ocr_mode = 'withMilliSec' # TODO: Change the value to 'withMilliSec' or 'withoutMilliSec'
                if ocr.need_pyt or framecount == 1:
                    time_ocr_frame = ocr.run_ocr((im[ocr_vertical_offset+self.main_config_dict['ocr_y_min']:ocr_vertical_offset+self.main_config_dict['ocr_y_max'], self.main_config_dict['ocr_x_min']:self.main_config_dict['ocr_x_max']], videoTimer), ocr_mode)
                else:
                    time_ocr_frame  = ocr.run_ocr(videoTimer, ocr_mode)
                if isinstance(time_ocr_frame, datetime):
                    date = time_ocr_frame.strftime("%d.%m.%Y")
                    time = time_ocr_frame.strftime("%H:%M:%S")
                    storing_output["Date"] = date
                    storing_output["Time"] = time
                    
                    if ocr_mode == 'withMilliSec':
                        millisec = int(time_ocr_frame.microsecond / 1000)
                        storing_output["Millisec"] = millisec
                    else:
                        storing_output["Millisec"] = np.nan
                    
                else:
                    storing_output["Date"] = np.nan
                    storing_output["Time"] = np.nan
                    storing_output["Millisec"] = np.nan
            else:
                    storing_output["Date"] = np.nan
                    storing_output["Time"] = np.nan
                    storing_output["Millisec"] = np.nan

            if self.traffic_light_info:
                # Traffic light state
                img_red = self.state.region_of_interest(im0, self.state.red_region)
                img_green = self.state.region_of_interest(im0, self.state.green_r
                    storing_output["Time"] = np.nan
                    storing_output["Millisec"] = np.nan
            else:
                    storing_output["Date"] = np.nan
                    storing_output["Time"] = np.nan
                    storing_output["Millisec"] = np.nan

            if self.traffic_light_info:
                # Traffic light state
                img_red = self.state.region_of_interest(im0, self.state.red_region)
                img_green = self.state.region_of_interest(im0, self.state.green_region)

                # Light Stateegion)

                # Light State
                current_state = self.state.light_state([img_red, img_green])
                storing_output['State'] = current_state

            # Image Preprocessing for inference
            t1 = time_sync()
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = self.model(im, augment=False, visualize=False)
            t3 = time_sync()
            dt[1] += t3 - t2
            if self.pt:
                pred = pred[0]

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)[0]
            t4 = time_sync()
            dt[2] += t4 - t3

            # Process predictions
            seen += 1

            s += '%gx%g ' % im.shape[2:]

            if len(pred):
                # Rescale boxes from img_size to im0 size
                pred[:, :4] = scale_coords(im.shape[2:], pred[:, :4], im0.shape).round()

                # Print results
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string 
        
            # Save the images or videos
            if self.inference_mode == 'SingleImage':
                self.frame = Visualize.drawBBOX(pred, im0)
                cv2.imwrite(self.output, self.frame)
            
            elif self.inference_mode == 'Video':
                # Update the tracker
                self.UpdateTracker(pred)

                # Storing values for post-processing
                if len(self.tracker) > 0:
                    output_data.extend(self.UpdateStorage_withTracker(storing_output))
                elif len(pred) > 0:
                    print("No Trackers")
                    output_data.extend(self.UpdateStorage_onlyYolo(storing_output, pred))
                else:
                    print("No Trackers/Predictions")
                    output_data.append(storing_output)

                t5 = time_sync()
                dt[3] += t5 - t4
                print(f'{s}Done. ({1/(t3 - t2):.3f}fps)(Post: {((t5 - t4)*1000):.3f}ms)')


                if self.save_infer_video:
                    if len(self.tracker) > 0:
                        frame = Visualize.drawTracker(self.tracker, im0, framecount)
                    elif len(pred) > 0:
                        frame = Visualize.drawBBOX(pred, im0, framecount)
                    else:
                        frame = Visualize.drawEmpty(im0, framecount)

                    if framecount == 1:  # new video
                        vid_path = self.output[:-4] + '_justInference' + self.output[-4:]
                        print(f"Video saving to : {vid_path}")
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
                    vid_writer.write(frame)      

        if self.inference_mode == 'Video' and self.save_infer_video:    
            vid_writer.release()

        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.3fms NMS per image at shape {(1, 3, *im.shape[2:])}, %.1fms Post-processing' % t)
        time_end = time_sync()
        print(f'Total time for inference (including pre and post-processing): {round(time_end-time_start, 2)}s')
        print(f'Average total fps: {round(framecount/round(time_end-time_start, 2), 2)}fps')
        
        df = pd.DataFrame(output_data)
        df.to_csv(f"{self.output_dir_path}/{str(self.file_stem_name)}_raw.csv")

    def parse_opt():
        parser = argparse.ArgumentParser()
        #parser.add_argument('--input', type=str, default=None, help=['path to input file(s)', '.MP4/.mkv/.png/.jpg/.jpeg'])
        parser.add_argument('--input', type=str, default='/home/mobilitylabextreme002/Desktop/Pixel Reading using Opencv/videos/short_video.mp4', help=['path to input file(s)', '.MP4/.mkv/.png/.jpg/.jpeg'])
        #parser.add_argument('--model_weights', type=str, default=None, help='model\'s weights path(s)')
        parser.add_argument('--model_weights', type=str, default='../weights/All_5_combined/weights/best.engine', help='model\'s weights path(s)')
        #parser.add_argument('--output', type=str, default=None, help=['path to save result(s)', '.MP4/.mkv/.png/.jpg/.jpeg'])
        parser.add_argument('--output', type=str, default='../test_for_light.mkv', help=['path to save result(s)', '.MP4/.mkv/.png/.jpg/.jpeg'])
        parser.add_argument('--minimap', default=False, action='store_true', help='Option to show the minimap in output -- True (or) False (default: False)')
        parser.add_argument('--trj_mode', default=False, action='store_true', help='Option to show trajectory in output -- True (or) False (default: False)')
        parser.add_argument('--imgSize','--img','--img_size', nargs='+', type=int, default=[1088, 1920], help='inference size h,w')
        parser.add_argument('--update_rate', type=int, default=30, help='Provide a number to update trajectory after certain frames')
        parser.add_argument('--disable_post_process', default=False, action='store_true', help='Disable Post-Processing (default: False)')
        parser.add_argument('--save_infer_video', default=False, action='store_true', help='Enable/Disable saving infer video before post-processing -- True (or) False (default: False if disable_post_process, otherwise True)')
        parser.add_argument('--save_class_frames', type=int, default=0, help='Save frames of requied class from 0 to 6 classes\
                                                    (0-Escooter, 1-Pedestrian, 2-Cyclist, 3-Motorcycle, 4-Car, 5-Truck, 6-Bus)')
        parser.add_argument('--traffic_light_info', type=bool, default=False, help="True to gather traffic light state information of the pedestrians crossing!")
        parser.add_argument("--light_region", type=list, default = [[350, 1504, 354, 1508], [358, 1504, 362, 1508], [217, 1464, 220, 1467], [224, 1463, 227, 1466]], help="coordinates of light region (Red and Green together) eg. [x, y, width, height]")         
        parser.add_argument("--night_mode", type=bool, default=False, help='detect in night video or day video eg. default is false detects for day light video')
        opt = parser.parse_args()
        #opt.imgSize *= 2 if len(opt.imgSize) == 1 else 1
        print_args(FILE.stem, opt)
        return opt

    def main(opt):
        Inference(**vars(opt))

    
if __name__ == "__main__":
    opt = Inference.parse_opt()
    fps = 30
    print("---- Traffic Camera Tracking (CARISSMA) ----")

    if opt.disable_post_process:
        opt.save_infer_video = True
    # setting limit for update_rate -> Number of times/s (Hz) [Example: 1 refers to 1 time per second. 30 refers to 30 times per second]
    if opt.update_rate > fps:
        opt.update_rate = 1
        print("[INFO] update_rate cannot exceed the video fps")
    elif opt.update_rate <= 0:
        opt.update_rate = fps
        print(f"[INFO] update_rate cannot be negative or 0.")
    opt.update_rate = int(fps/opt.update_rate)
    print(f"[INFO] update_rate is set to every {opt.update_rate} frame")

    Inference.main(opt)
    print("\n")
    if not opt.disable_post_process:
        __output_path_processing = Path(opt.output)
        file_stem_name = __output_path_processing.stem
        parent_directory = __output_path_processing.parents[0]
        output_dir_path = parent_directory/file_stem_name

        print("---- Post-Processing ----")
        post = PostProcess(f"{output_dir_path}/{str(file_stem_name)}_raw.csv", opt.input,
                        opt.output, opt.minimap, opt.trj_mode, opt.update_rate, opt.save_class_frames)

        post.run()