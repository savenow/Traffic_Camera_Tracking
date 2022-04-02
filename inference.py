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

from extract_stored_detections_copy import PostProcess

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class Inference():
    def __init__(self, input, model_weights, output=None, trj_output=None, 
                minimap=False, trj_mode=False,  imgSize=[1920, 1920],
                update_rate = 30):        
        # Inference Params
        self.img_size = imgSize
        self.conf_thres = 0.8
        self.iou_thres = 0.5
        self.agnostic_nms = False
        self.max_det = 1000
        self.classes = None # Filter classes

        self.device = torch.device('cuda:0')
        self.half = True
        cudnn.benchmark = True
        self.update_rate = update_rate
        
        # Checking input
        if os.path.isfile(input):
            # Further functionality needs to be added for Folder Inference :))
            if input[-4:] in ['.png', '.jpg']:
               self.input = input
               self.inference_mode = 'SingleImage'
            elif input[-4:] in ['.mp4', '.mkv', '.avi']:
                self.input = input
                self.inference_mode = 'Video'
                self.fps = 30
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
            self.file_stem_name = self.output.split('/')[-1][:-4]
        
        if trj_output == None:
            self.trajectory_output = self.input.split('/')[-1]
        else:
            self.trajectory_output = trj_output

        self.showTrajectory = trj_mode

        # setting limit for update_rate -> Number of times/s (Hz) [Example: 1 refers to 1 time per second. 30 refers to 30 times per second]
        if self.update_rate > self.fps:
            self.update_rate = 1
            print("[INFO] update_rate cannot exceed the video fps")
        elif self.update_rate <= 0:
            self.update_rate = self.fps
            print(f"[INFO] update_rate cannot be negative or 0.")
        self.update_rate = int(self.fps/self.update_rate)
        print(f"[INFO] update_rate is set to every {self.update_rate} frame")

        self.trajectory_retain_duration = 100 # Number of frames the trajectory for each tracker id must be retained before removal

        # Loading Model
        model = DetectMultiBackend(self.model_weights, device=self.device, dnn=None)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
        self.names = ['Escooter', 'Pedestrian', 'Cyclist']
        
        if self.pt:
            model = model.model.half()
        self.model = model

        # Initialize Tracker
        self.Objtracker = Sort(max_age=40, min_hits=7, iou_threshold=0.3)
        self.Objtracker.reset_count()

        # Camera Calibration data: Used for velocity estimation
        self.enable_minimap = minimap
        self.enable_trajectory = trj_mode
        self.Calib = Calibration()
        
        # Parameters for velocity estimation
        self.trackDict = defaultdict(list)

        # For storing conversion values
        self.Minimap_storage = Minimap()

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
            center_x = (detection[0] + detection[2])/2        
            _, max_y = sorted((detection[1], detection[3]))

            trackID = int(detection[9])
            self.trackDict[trackID].append((int(center_x), int(max_y)))
            
            if len(self.trackDict[trackID]) > 2: 
                previous_point = self.Calib.projection_pixel_to_world(self.trackDict[trackID][-2])
                current_point = self.Calib.projection_pixel_to_world(self.trackDict[trackID][-1])

                del self.trackDict[trackID][-2]

                distance_metres = round(float(math.sqrt(math.pow(previous_point[0] - current_point[0], 2) + math.pow(previous_point[1] - current_point[1], 2))), 2)
                speed_kmH = round(float(distance_metres * self.fps * 3.6), 2)
                output_array = np.append(detection, speed_kmH)
                velocity_array.append(output_array)

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
        ocr_vertical_offset = int((1920-1080)/2) # Since the imgSize for inference is 1920x1920 and input video is 1920x1080, some padding is automatically applied by Yolo. Offsetting the y-values for this padding.
        output_data = [] # For writing detection/tracker data to .csv for post processing
        Visualize = Visualizer(self.enable_minimap, self.enable_trajectory, self.update_rate, self.trajectory_retain_duration)
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        framecount = 0
        time_start = time_sync()
        for path, im, im0, vid_cap, s, videoTimer in dataset:
            framecount += 1
            if framecount < -1:
                continue
            elif framecount > 73000:
            # elif framecount > 100:
                vid_writer.release()
                break
            storing_output = {}
            storing_output["Video_Internal_Timer"]= videoTimer
            # OCR Reading Timestamp
            if ocr.need_pyt or framecount == 1:
                time_ocr_frame = ocr.run_ocr((im[ocr_vertical_offset+4:ocr_vertical_offset+41, 0:568], videoTimer))
            else:
                time_ocr_frame  = ocr.run_ocr(videoTimer)

            if isinstance(time_ocr_frame, datetime):
                date = time_ocr_frame.strftime("%d.%m.%Y")
                time = time_ocr_frame.strftime("%H:%M:%S")
                millisec = int(time_ocr_frame.microsecond / 1000)
                storing_output["Date"] = date
                storing_output["Time"] = time
                storing_output["Millisec"] = millisec
            else:
                storing_output["Date"] = np.nan
                storing_output["Time"] = np.nan
                storing_output["Millisec"] = np.nan

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

                # Velocity Estimation
                velocity_estimation = []
                calculated_velocity = self.VelocityEstimation(velocity_estimation)

                if calculated_velocity:
                    frame = Visualize.drawAll(calculated_velocity, im0, framecount)
                elif len(self.tracker) > 0:
                    frame = Visualize.drawTracker(self.tracker, im0, framecount)
                elif len(pred) > 0:
                    frame = Visualize.drawBBOX(pred, im0, framecount)
                else:
                    frame = Visualize.drawEmpty(im0, framecount)
                   
                t5 = time_sync()
                dt[3] += t5 - t4
                print(f'{s}Done. ({1/(t3 - t2):.3f}fps)(Post: {((t5 - t4)*1000):.3f}ms)')

                # if vid_path != self.output:  # new video
                #     vid_path = self.output
                #     w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #     h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #     vid_writer = cv2.VideoWriter(self.output, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (w, h))
                # vid_writer.write(frame)      

        # if self.inference_mode == 'Video':    
        #     vid_writer.release()
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.3fms NMS per image at shape {(1, 3, *im.shape[2:])}, %.1fms Post-processing' % t)
        time_end = time_sync()
        print(f'Total time for inference (including pre and post-processing): {round(time_end-time_start, 2)}s')
        print(f'Average total fps: {round(framecount/round(time_end-time_start, 2), 2)}fps')
        
        df = pd.DataFrame(output_data)
        df.to_csv(f"{self.file_stem_name}.csv")


    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str, default=None, help=['path to input file(s)', '.MP4/.mkv/.png/.jpg/.jpeg'])
        parser.add_argument('--model_weights', type=str, default=None, help='model\'s weights path(s)')
        parser.add_argument('--output', type=str, default=None, help=['path to save result(s)', '.MP4/.mkv/.png/.jpg/.jpeg'])
        parser.add_argument('--trj_output', type=str, default=None, help=['path to save the trjectory result(s)', '.png/.jpg/.jpeg'])
        parser.add_argument('--minimap', default=False, action='store_true', help='provied option for showing the minimap in result -- True (or) False')
        parser.add_argument('--trj_mode', default=False, action='store_true', help='provied option to turn on or off the trjectory recording -- True (or) False')
        parser.add_argument('--imgSize','--img','--img_size', nargs='+', type=int, default=[1920], help='inference size h,w')
        parser.add_argument('--update_rate', type=int, default=30, help='provide a number to update a trajectory after certain frames')
        opt = parser.parse_args()
        opt.imgSize *= 2 if len(opt.imgSize) == 1 else 1
        print_args(FILE.stem, opt)
        return opt

    def main(opt):
        Inference(**vars(opt))

    
if __name__ == "__main__":
    opt = Inference.parse_opt()
    print("---- Traffic Camera Tracking (CARISSMA) ----")
    Inference.main(opt)
    time.sleep(5.0)
    print(" ")
    print("---- Post-Processing ----")
    post = PostProcess(f"{opt.output.split('/')[-1][:-4]}.csv", opt.input,
                       opt.output, opt.minimap, opt.trj_mode, opt.update_rate)

    post.run()