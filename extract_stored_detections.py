import numpy as np
import pandas as pd
import cv2
import sys
from pathlib import Path
import os
import argparse

from visualizer import Visualizer, Minimap
from calibration import Calibration

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class ExtractFromCSV():
    def __init__(self, data_file, input_video, output_video, enable_minimap, enable_trj_mode):
        self.detections_dataframe = pd.read_csv(data_file)

        self.video_cap = cv2.VideoCapture(input_video)

        frame_width = int(self.video_cap.get(3))
        frame_height = int(self.video_cap.get(4))
        self.video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))
        self.Visualize = Visualizer(enable_minimap, enable_trj_mode) 

    def group_by_internalTimer(self):
        framecounter = 0
        prev_frametime = -1
        list_grouped_by_frametimes = []
        detection_per_frame = []

        for index, row in self.detections_dataframe.iterrows():
            df_frametime = row['Video_Internal_Timer']
            if prev_frametime == -1: # To properly allocate df_frametime with the framecount. Visualizer functions depends on framecount.
                prev_frametime = df_frametime
                detection_per_frame.append(row)
                framecounter += 1
            elif prev_frametime != df_frametime:
                framecounter += 1
                list_grouped_by_frametimes.append(detection_per_frame.copy())
                detection_per_frame = []
                detection_per_frame.append(row)
                prev_frametime = df_frametime
            else:
                detection_per_frame.append(row)
        print("[INFO] Finished grouping the pandas dataframe by frametime")
        return list_grouped_by_frametimes
        

    def groupedData_toVideoWriter(self, list_grouped_by_frametimes):
        framecounter = 0
        if self.video_cap.isOpened():
            for data in list_grouped_by_frametimes:
                df_frametime = data[0]['Video_Internal_Timer']
                self.video_cap.set(cv2.CAP_PROP_POS_MSEC, df_frametime)
                ret, frame = self.video_cap.read()
                if ret:
                    framecounter += 1
                    print(f"[INFO] Processing frame {framecounter}")
                    font = cv2.FONT_HERSHEY_SIMPLEX
                
                    # Timestamp-Text properties
                    org = (700, 30)
                    fontScale = 1
                    color = (0, 0, 255)
                    thickness = 2
                    
                    # Displaying the extracted timestamp
                    image = cv2.putText(frame, f"Extracted Time: {data[0]['Time']} {data[0]['Millisec']}", org, font, 
                                fontScale, color, thickness, cv2.LINE_AA)

                    for detection in data:
                        if not pd.isna(detection['Tracker_ID']):
                            # Drawing Trackers
                            outer_array = []
                            detection_array = []
                            x1, y1 = detection['BBOX_TopLeft'][1:-1].split(',')
                            x2, y2 = detection['BBOX_BottomRight'][1:-1].split(',')
                            detection_array.append(int(x1))
                            detection_array.append(int(y1))
                            detection_array.append(int(x2))
                            detection_array.append(int(y2))
                            detection_array.append(detection['Conf_Score']/100)
                            detection_array.append(detection['Class_ID'])
                            detection_array.extend([0, 0, 0]) # Placeholder values. The visualizer function doesn't need these but kept in places to align with the indices.
                            detection_array.append(detection['Tracker_ID'])
                            outer_array.append(detection_array)
                            image = self.Visualize.drawTracker(outer_array, image, framecounter)
                        elif not pd.isna(detection['Class_ID']):
                            # Drawing just BBOXes
                            outer_array = []
                            detection_array = []
                            x1, y1 = detection['BBOX_TopLeft'][1:-1].split(',')
                            x2, y2 = detection['BBOX_BottomRight'][1:-1].split(',')
                            detection_array.append(int(x1))
                            detection_array.append(int(y1))
                            detection_array.append(int(x2))
                            detection_array.append(int(y2))
                            detection_array.append(detection['Conf_Score']/100)
                            detection_array.append(detection['Class_ID'])
                            outer_array.append(detection_array)
                            image = self.Visualize.drawBBOX(outer_array, image, framecounter)
                        else:
                            # No Detections/Trackers. Just drawing the minimap (if enabled)
                            image = self.Visualize.drawEmpty(image, framecounter)
                    
                    self.video_writer.write(image)
                    # if framecounter > 200:
                    #     break
                else:
                    break

        print(f'Total Number of frames: {framecounter}')

        self.video_writer.release()
        self.video_cap.release()

    def run(self):
        frametime_group_list = self.group_by_internalTimer()
        self.groupedData_toVideoWriter(frametime_group_list)

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default=None, help=['path to .csv data'])
    parser.add_argument('--input_video', type=str, default=None, help=['input video file, should be the same one used in .csv'])
    parser.add_argument('--output_video', type=str, default=None, help=['path to save result(s)'])
    parser.add_argument('--enable_minimap', default=True, action='store_true', help='provied option for showing the minimap in result -- True (or) False')
    parser.add_argument('--enable_trj_mode', default=True, action='store_true', help='provied option to turn on or off the trjectory recording -- True (or) False')
    opt = parser.parse_args()
    print("---- Traffic Camera Tracking (CARISSMA) ----")
    print("---- Post-Processing ----")
    return opt

if __name__ == '__main__':
    opt = parser_opt()
    obj = ExtractFromCSV(**vars(opt))
    obj.run()