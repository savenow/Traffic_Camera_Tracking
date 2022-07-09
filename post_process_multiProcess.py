from cmath import nan
import numpy as np
import pandas as pd
import cv2
import sys
from pathlib import Path
import os
import argparse
import math
from matplotlib import pyplot as plt
import warnings
from torch import det
warnings.filterwarnings("ignore") # To ignore certain warnings from Pandas
from tqdm.auto import tqdm

from visualizer import Visualizer, Minimap
from calibration import Calibration
from imutils.video import FPS
from collections import namedtuple, defaultdict
from heading_angle import Angle
from kalmanfilter import KalmanFilter
import subprocess as sp
import multiprocessing as mp
from os import remove
import time
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class PostProcess():
    def __init__(self, data_file, input_video, output_video, enable_minimap, enable_trj_mode, trajectory_update_rate, save_class_frames):
        self.detections_dataframe = pd.read_csv(data_file, index_col=[0])
        __output_video_original_path = Path(output_video)
        self.file_name = __output_video_original_path.stem
        self.parent_directory = __output_video_original_path.parents[0]
        self.output_directory = self.parent_directory / self.file_name
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        if not os.path.exists(self.output_directory/"Save-frames"):
            os.makedirs(self.output_directory/"Save-frames")
        else:
            shutil.rmtree(self.output_directory/"Save-frames")           # Removes all the subdirectories!
            os.makedirs(self.output_directory/"Save-frames")

        self.detections_dataframe = self.removeErrorTimers(self.detections_dataframe)
        self.input_video = input_video
        self.outputfile_name = self.output_directory / __output_video_original_path.name
        self.video_fps = 30
        self.num_processes = int(mp.cpu_count() * 0.4)
        self.trajectory_retain_duration = 250
        self.Visualize = Visualizer(enable_minimap, enable_trj_mode, trajectory_update_rate, self.trajectory_retain_duration, save_class_frames)
        self.trackDict = defaultdict(list)
        self.angleDict = defaultdict(list)
        self.kf = KalmanFilter()
        self.angle = Angle()
        

    def removeErrorTimers(self, df):
        # At the end of .csv file, some erroneous data from tracker has been found with Video_Internal_Timer = 0. The following code snippet would remove them from the dataframe
        last_row_df = df.shape[0] - 1
        if df.iloc[last_row_df]['Video_Internal_Timer'] == 0:
            last_index_drop = last_row_df + 1
            df_rev = df[::-1]
            start_index_drop = -1    
            for row, col in df_rev.iterrows():
                if col['Video_Internal_Timer'] != 0:
                    start_index_drop = row + 1
                    break
            df.drop(df.index[start_index_drop:last_index_drop], 0, inplace=True)
        return df

    def group_by_internalTimer(self, df):
        vid_timer_gb = df.groupby(by=['Video_Internal_Timer'])
        unique_vid_timer = df.Video_Internal_Timer.unique()
        list_grouped_by_frametimes = []
        
        for vid_timer in unique_vid_timer:
            g = vid_timer_gb.get_group(vid_timer)
            ls = []
            for index, row in g.iterrows():
                ls.append(row)
            list_grouped_by_frametimes.append(ls)

        print("[INFO] Finished grouping the pandas dataframe by frametime")
        return list_grouped_by_frametimes
    
    def group_by_internalTimer_with_index(self, df):
        vid_timer_gb = df.groupby(by=['Video_Internal_Timer'])
        unique_vid_timer = df.Video_Internal_Timer.unique()
        list_grouped_by_frametimes = []
        
        for vid_timer in unique_vid_timer:
            g = vid_timer_gb.get_group(vid_timer)
            ls = []
            for index, row in g.iterrows():
                ls.append([index, row])
            list_grouped_by_frametimes.append(ls)

        return list_grouped_by_frametimes
    
    def groupedData_toVideoWriter(self, num_processes):
        framecounter = 0
        self.video_cap = cv2.VideoCapture(self.input_video)
        self.frame_width = int(self.video_cap.get(3))
        self.frame_height = int(self.video_cap.get(4))
        self.video_writer = cv2.VideoWriter()
        min_vid_timer = int(self.final_df['Video_Internal_Timer'].min())
        max_vid_timer = int(self.final_df['Video_Internal_Timer'].max())
        
        total_frames = int((max_vid_timer - min_vid_timer)/(1000/self.video_fps)) + 1
        frame_jump_unit =  total_frames// self.num_processes
        self.video_cap.set(cv2.CAP_PROP_POS_MSEC, min_vid_timer)
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_jump_unit * num_processes)
        self.video_writer.open("output_{}.mp4".format(num_processes), cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.frame_width,self.frame_height), True)
        pbar = tqdm(total=frame_jump_unit, leave=False, bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}')

        try:
            while framecounter < frame_jump_unit:
                ret, frame = self.video_cap.read()
                if ret:
                    vid_timer = int(self.video_cap.get(cv2.CAP_PROP_POS_MSEC))
                
                    if vid_timer > max_vid_timer:
                        break

                    pbar.update(1)
                    
                    for data in self.groupedByFrametime:
                        df_frametime = data[0]['Video_Internal_Timer']
                        
                        # Checking for interal_timer from .csv file and matching it with the internal timer from video file (For syncing frames)
                        if df_frametime == vid_timer:
                            framecounter += 1
                            #print(f"[INFO] Saving frame {framecounter}")
                            outer_array = []
                            for detection in data:
                                if not pd.isna(detection['Speed']):
                                    # Drawing Trackers
                                    detection_array = []
                                    x1 = detection['BBOX_TopLeft_x']
                                    y1 = detection['BBOX_TopLeft_y']
                                    x2 = detection['BBOX_BottomRight_x']
                                    y2 = detection['BBOX_BottomRight_y']
                                    center_x = int((int(x1)+int(x2))/2)
                                    if detection['Class_ID'] in [0,1,2]: 
                                        _, center_y = sorted((int(y1),int(y2)))
                                    elif detection['Class_ID'] in [3,4,5,6]:
                                        center_y = (int(y1) + int(y2))/2
                                    trk_id = int(detection['Tracker_ID'])
                                    self.trackDict[trk_id].append((int(center_x),int(center_y)))
                                    detection_array.append(int(x1))
                                    detection_array.append(int(y1))
                                    detection_array.append(int(x2))
                                    detection_array.append(int(y2))
                                    if not pd.isna(detection['Conf_Score']):
                                        detection_array.append(detection['Conf_Score']/100)    
                                    else:
                                        detection_array.append(-1)
                                    detection_array.append(detection['Class_ID'])
                                    detection_array.extend([0, 0, 0]) # Placeholder values. The visualizer function doesn't need these but kept in places to align with the indices.
                                    detection_array.append(detection['Tracker_ID'])
                                    detection_array.append(detection['Speed'])
                                    detection_array.append(detection['Arrow_points'][0])
                                    detection_array.append(detection['Arrow_points'][1])
                                    if len(self.trackDict[trk_id])>10:
                                        detection_array.append(self.trackDict)
                                        del self.trackDict[trk_id][0]
                                    outer_array.append(detection_array)


                                elif not pd.isna(detection['Tracker_ID']):
                                    # Drawing Trackers
                                    detection_array = []
                                    x1 = detection['BBOX_TopLeft_x']
                                    y1 = detection['BBOX_TopLeft_y']
                                    x2 = detection['BBOX_BottomRight_x']
                                    y2 = detection['BBOX_BottomRight_y']
                                    detection_array.append(int(x1))
                                    detection_array.append(int(y1))
                                    detection_array.append(int(x2))
                                    detection_array.append(int(y2))
                                    if not pd.isna(detection['Conf_Score']):
                                        detection_array.append(detection['Conf_Score']/100)    
                                    else:
                                        detection_array.append(-1)
                                    detection_array.append(detection['Class_ID'])
                                    detection_array.extend([0, 0, 0]) # Placeholder values. The visualizer function doesn't need these but kept in places to align with the indices.
                                    detection_array.append(detection['Tracker_ID'])
                                    detection_array.extend([0.0])
                                    detection_array.extend([0, 0, 0])
                                    outer_array.append(detection_array)

                                    
                                elif not pd.isna(detection['Class_ID']):
                                    # Drawing just BBOXes
                                    detection_array = []
                                    x1 = detection['BBOX_TopLeft_x']
                                    y1 = detection['BBOX_TopLeft_y']
                                    x2 = detection['BBOX_BottomRight_x']
                                    y2 = detection['BBOX_BottomRight_y']
                                    detection_array.append(int(x1))
                                    detection_array.append(int(y1))
                                    detection_array.append(int(x2))
                                    detection_array.append(int(y2))
                                    detection_array.append(detection['Conf_Score']/100)
                                    detection_array.append(detection['Class_ID'])
                                    outer_array.append(detection_array)

                                else:
                                    # No Detections/Trackers. Just drawing the minimap (if enabled)
                                    image = self.Visualize.drawEmpty(frame, framecounter)

                            image = self.Visualize.drawAll(outer_array, frame, framecounter, self.output_directory)
                            self.video_writer.write(image)

                else:
                    break
        except:
            # Release resources
            self.video_cap.release()
            self.video_writer.release() 

        # Release resources
        pbar.close()
        self.video_cap.release()
        self.video_writer.release()        

    def Save_angle_to_csv(self, df_with_index, final_df):
        last_df = final_df.copy()
        outer_array = []
        for data in df_with_index:
            for detection in data:
                if not pd.isna(detection[1]['Speed']):
                    index = detection[0]
                    frame_time = detection[1]['Video_Internal_Timer']
                    x1 = detection[1]['BBOX_TopLeft_x']
                    y1 = detection[1]['BBOX_TopLeft_y']
                    x2 = detection[1]['BBOX_BottomRight_x']
                    y2 = detection[1]['BBOX_BottomRight_y']
                    center_x = int((int(x1)+int(x2))/2)
                    if detection[1]['Class_ID'] in [0,1,2]: 
                        _, center_y = sorted((int(y1),int(y2)))
                    elif detection[1]['Class_ID'] in [3,4,5,6]:
                        center_y = (int(y1) + int(y2))/2
                    trk_id = int(detection[1]['Tracker_ID'])
                    cx1 = int(float(detection[1]['Arrow_points'][0]))
                    cy1 = int(float(detection[1]['Arrow_points'][1]))
                    self.angleDict[trk_id].append((int(center_x),int(center_y)))

                    new_row = {
                                'Index': index,
                                'Video_Internal_Timer': int(frame_time), 
                                'Heading_angle': np.nan
                                }

                    points_ = [[cx1, cy1], [int(center_x), int(center_y)], [x2, cy1]]
                    if len(self.angleDict[trk_id])<=10:
                        if detection[1]['Speed']!= 0:
                            angle_ = self.angle.findangle(points=points_)

                            new_row = {
                                'Index': index,
                                'Video_Internal_Timer': int(frame_time), 
                                'Heading_angle': angle_
                                } 

                    elif len(self.angleDict[trk_id])>10:
                        for pt in self.angleDict[trk_id]:
                            predicted = self.kf.predict(pt[0], pt[1])
                        del self.angleDict[trk_id][-1]

                        pred = predicted
                        for i in range(2):
                            pred = self.kf.predict(pred[0], pred[1])

                        points = [[cx1, cy1], [int(pred[0]), int(pred[1])], [x2, cy1]]
                        if detection[1]['Speed']!= 0:
                            angle = self.angle.findangle(points=points)

                            new_row = {
                                'Index': index,
                                'Video_Internal_Timer': int(frame_time), 
                                'Heading_angle': angle
                                } 

                    outer_array.append(new_row)

        df_angle = pd.DataFrame(outer_array)

        # Copy Angle values to the original dataframe by matching index values
        for i in df_angle['Index']:
            last_df.loc[i, 'Heading_angle'] = df_angle.loc[df_angle['Index']==i, 'Heading_angle'].values[0]
        
        return last_df
    
    def combine_output_files(self, num_processes):
        # Create a list of output files and store the file names in a txt file
        list_of_output_files = ["output_{}.mp4".format(i) for i in range(num_processes)]
        with open("list_of_output_files.txt", "w") as f:
            for t in list_of_output_files:
                f.write("file {} \n".format(t))

        # use ffmpeg to combine the video output files
        ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec nvenc_hevc " + str(self.outputfile_name)
        #ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + str(self.outputfile_name)
        sp.Popen(ffmpeg_cmd, shell=True).wait()

        # Remove the temperory output files
        for f in list_of_output_files:
            remove(f)
        remove("list_of_output_files.txt")
    
    def get_video_frame_details(self, file_name):
        cap = cv2.VideoCapture(file_name)

        # get height, width and frame count of the video
        width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        return width, height, frame_count

    def multi_process(self):
        width, height, frame_count = self.get_video_frame_details(self.input_video)
        print("[INFO] Video processing using {} processes...".format(self.num_processes))
        
        start_time = time.time()

        # Parallel execution of a function across multiple input values
        p = mp.Pool(self.num_processes)
        
        p.map(self.groupedData_toVideoWriter, range(self.num_processes))

        self.combine_output_files(self.num_processes)

        end_time = time.time()

        total_processing_time = end_time - start_time
        print("\n[INFO] Finished saving post-processed video !!")
        print("[INFO] Elasped time: {:.2f}s".format(total_processing_time))
        print("[INFO] Approx. FPS: {:.2f}".format(frame_count/total_processing_time))

    # Helper functions for step 1: tracker removal
    def get_tracker_IDs(self, dataframe):
        """ This function helps to get available tracker IDs in the dataframe
            returns the List of Tracker IDs
        """
        tracker_ids = dataframe.loc[pd.isna(dataframe['Tracker_ID'])==False, 'Tracker_ID']
        tracker_ids = pd.unique(tracker_ids)
        tracker_ids = sorted(tracker_ids)

        return tracker_ids

    def get_arranged_IDs(self, dataframe, tracker_ids):
        """ This function helps to rearrange available tracker IDs in the dataframe in correct oreder of numbers(e.g. 1,2,3,4,..etc)
            returns the arranged dataframe and the List of arranged Tracker IDs
            Args:-
            tracker_ids = List of tracker IDs got from the 'get_tracker_IDs()' function.
        """

        ls = [i+1 for i in range(len(tracker_ids))]

        for i in range(len(dataframe['Tracker_ID'])):
            for j in range(len(tracker_ids)):
                if dataframe.loc[i, 'Tracker_ID'] == tracker_ids[j]:
                    dataframe.loc[i, 'Tracker_ID'] = ls[j]

        return dataframe, ls

    def get_tracker_Id_counts(self, dataframe, tracker_ids, arranged_ids):
        """ This function helps to get the dictionary which contains Tracker_IDs as a key and it's total counts in the dataframe as a value
            returns Dictionary of total counts of each Tracker_IDs
            Args:-
            tracker_ids = List of tracker IDs got from the 'get_tracker_IDs()' function.
            arranged_ids = List of arranged tracker IDs got from the 'get_arranged_IDs()' function.
        """
        var_id={}
        for i in range(len(tracker_ids)):
            var_id[i+1] = dataframe.loc[dataframe['Tracker_ID'] == arranged_ids[i], 'Tracker_ID']

        id_len = {}
        count = 0
        for key, values in list(var_id.items()):
            id_len[int(arranged_ids[count])] = len(var_id[key])
            count += 1

        return id_len

    def get_ids_to_delete(self, id_len, minimum_id_count):
        """ This function helps to the List of tracker_ids which are going to be deleted from the dataframe
            returns the List of tracker_ids to be deleted 
            Args:-
            id_len = Dictionay of total count got from the 'get_tracker_Id_counts()' function.
            minimum_id_count = a minimum number of ID_counts to decide the tracker_id to delete.  
        """
        id_to_delete = [float(k) for k,v in id_len.items() if v < minimum_id_count]
        return id_to_delete

    def get_index_to_delete(self, dataframe, id_to_delete):
        """ This function helps to the List of indexes of the daraframe which are going to be deleted.
            returns the List of indexes to be deleted 
            Args:-
            id_to_delete = List of tracker_ids to be deleted got from the 'get_ids_to_delete()' function.
        """
        list_to_delete = []
        for i in range(len(dataframe['Tracker_ID'])):
            if dataframe.loc[i, 'Tracker_ID'] in id_to_delete:
                list_to_delete.append(i)
        return list_to_delete

    def set_index_nan(self, dataframe, list_to_delete):
        """ This function helps to delete the selected rows from the dataframe
            returns the new datataframe after droping the selected rows.
            Args:-
            list_to_delete = the list of indexes which are used to delete the rows from the dataframe 
        """
        for i in range(len(dataframe['Tracker_ID'])):
            if i in list_to_delete:
                dataframe.iloc[i, 4:] = np.nan

        return dataframe

    # Helper function for step 2: interpolation
    def find_missing_vidTimer(self, df_duplicate, vidTimer_present_in_interpolated_df):
        """
        There could also by some frames where no trackers are present, but these frames cannot be omitted from the Dataframe, as our implementation of VideoWriter requires a value for each frame.
        In the following code snippet, the interpolated_df is searched for such missing frames and these missing frames are filled back into the Dataframe with empty values.

        Args:
            df_duplicate (pd.Dataframe): Duplicate of the original .csv Dataframe
            vidTimer_present_in_interpolated_df (List): Video_Internal_Timer present in interpolated data

        Returns:
            missing_vidTimer (pd.Dataframe): Dataframe containing all rows from the original .csv which are not present in the interpolated data
        """
        video_timer_df = df_duplicate['Video_Internal_Timer'].unique()
        missing_vidTimer_list = []
        for video_actual_timer in video_timer_df:
            if video_actual_timer not in vidTimer_present_in_interpolated_df:
                idx = df_duplicate.index[df_duplicate['Video_Internal_Timer'] == video_actual_timer][0]
                row = df_duplicate.iloc[idx]
                new_row = {
                    'Video_Internal_Timer': video_actual_timer, 
                    'Date': row['Date'], 'Time': row['Time'], 'Millisec': row['Millisec'], 'Tracker_ID': np.nan, 
                    'Class_ID': np.nan, 'Conf_Score': np.nan, 'BBOX_TopLeft_x': np.nan, 'BBOX_TopLeft_y': np.nan,
                    'BBOX_BottomRight_x': np.nan, 'BBOX_BottomRight_y': np.nan
                }
                missing_vidTimer_list.append(new_row)

        missing_vidTimer = pd.DataFrame(missing_vidTimer_list)
        return missing_vidTimer

    # Main Post-processing functions
    def remove_tracker(self, dataframe):
        tracker_ids = self.get_tracker_IDs(dataframe)
        dataframe, arranged_ids = self.get_arranged_IDs(dataframe, tracker_ids)
        id_counts = self.get_tracker_Id_counts(dataframe, tracker_ids, arranged_ids)
        id_to_delete = self.get_ids_to_delete(id_counts, 80)
        list_to_delete = self.get_index_to_delete(dataframe, id_to_delete)
        new_dataframe = self.set_index_nan(dataframe, list_to_delete)
        ids_after_set_nan = self.get_tracker_IDs(new_dataframe)
        final_dataframe, id_latest = self.get_arranged_IDs(new_dataframe, ids_after_set_nan)
        return final_dataframe

    def interpolate_data(self, dataframe):
        """
        During the entire duration of each tracker_id, there could be frames inbetween where they are not present. In this function, we are grouping the .csv by Tracker_ID and interpolating the missing values
        for the BBOX Coordinates. Date/Time for each missing row is copied from the originial Dataframe and the Class_ID / Conf_Thres are just left empty. Quadratic Interpolation is used by default and if the tracker has only two rows of BBOX
        Coordinates available, then Linear Interpolation is used. Also the rows where no tracker is present are also accordingly handled.

        Args:
            dataframe (pd.Dataframe): Dataframe object without all the unwanted trackers removed
        Returns:
            interpolated_final_df (pd.Dataframe): Dataframe object with all the missing tracker coordinates interpolated
        """
        unique_trackers = dataframe.Tracker_ID.unique()
        tracker_group = dataframe.groupby('Tracker_ID')

        video_timer_df = dataframe['Video_Internal_Timer'].unique()
        interpolated_df_final_list = []
        vidTimer_present_in_interpolated_df = []
        for unique_tracker_id in unique_trackers: # Looping for each unique Tracker_ID
            if not pd.isna(unique_tracker_id):
                single_tracker_group = tracker_group.get_group(unique_tracker_id).reset_index(drop=True)
                
                # Getting minimum and maximum 'Video_Internal_Timer' during which the Tracker_ID is present.
                min_vid_timer = single_tracker_group['Video_Internal_Timer'].min()
                max_vid_timer = single_tracker_group['Video_Internal_Timer'].max()

                final_list_tracker_id = []
                for index in range(np.where(video_timer_df == min_vid_timer)[0][0], np.where(video_timer_df == max_vid_timer)[0][0] + 1):
                    if video_timer_df[index] not in single_tracker_group['Video_Internal_Timer'].values:
                        # If Tracker_ID is not present in this specific Video_Internal_Timer, creating a new row by having default values as None
                        # These missing Tracker_ID are later interpolated
                        idx = dataframe.index[dataframe['Video_Internal_Timer'] == video_timer_df[index]][0]
                        row_tracker_id = dataframe.iloc[idx]
                        new_row = {
                            'Video_Internal_Timer': video_timer_df[index], 
                            'Date': row_tracker_id['Date'], 'Time': row_tracker_id['Time'], 'Millisec': row_tracker_id['Millisec'], 'Tracker_ID': unique_tracker_id, 
                            'Class_ID': np.nan, 'Conf_Score': np.nan, 'BBOX_TopLeft_x': np.nan, 'BBOX_TopLeft_y': np.nan,
                            'BBOX_BottomRight_x': np.nan, 'BBOX_BottomRight_y': np.nan
                        }
                        final_list_tracker_id.append(new_row)
                    else:
                        # If Tracker_ID is present, just copying the row.
                        idx = single_tracker_group.index[single_tracker_group['Video_Internal_Timer'] == video_timer_df[index]][0]
                        row_tracker_id = single_tracker_group.iloc[idx]
                        x1, y1 = row_tracker_id['BBOX_TopLeft'][1:-1].split(',')
                        x2, y2 = row_tracker_id['BBOX_BottomRight'][1:-1].split(',')
                        try:
                            x1 = int(x1)
                            y1 = int(y1)        
                            x2 = int(x2)
                            y2 = int(y2)
                        except ValueError:
                            print(f'{x1}, {y1}, {x2}, {y2}')
                            print(row_tracker_id['BBOX_TopLeft'])
                            print(row_tracker_id['BBOX_BottomRight'])
                            print(row_tracker_id['BBOX_TopLeft'][1:-1].split(','))
                            print(row_tracker_id['BBOX_BottomRight'][1:-1].split(','))
                            raise ValueError

                        new_row = {
                            'Video_Internal_Timer': video_timer_df[index], 
                            'Date': row_tracker_id['Date'], 'Time': row_tracker_id['Time'], 'Millisec': row_tracker_id['Millisec'], 'Tracker_ID': unique_tracker_id, 
                            'Class_ID': row_tracker_id['Class_ID'], 'Conf_Score': row_tracker_id['Conf_Score'], 'BBOX_TopLeft_x': x1, 'BBOX_TopLeft_y': y1,
                            'BBOX_BottomRight_x': x2, 'BBOX_BottomRight_y': y2
                        }
                        final_list_tracker_id.append(new_row)
                    vidTimer_present_in_interpolated_df.append(video_timer_df[index])

                # Converting the list of rows to Dataframe and interpolating the missing BBOX_Values.
                df_new_tracker = pd.DataFrame(final_list_tracker_id)
                df_new_tracker = df_new_tracker.sort_values(by=['Video_Internal_Timer']).reset_index(drop=True)
                try:
                    # Using Quadratic Interpolation by default. Works only if more than two BBOX coordinates are associated with the tracker
                    bbox_position = df_new_tracker[['BBOX_TopLeft_x', 'BBOX_TopLeft_y', 'BBOX_BottomRight_x', 'BBOX_BottomRight_y']].interpolate(method='quadratic', axis=0)
                except ValueError:
                    # If just two points are present, then using linear interpolation
                    bbox_position = df_new_tracker[['BBOX_TopLeft_x', 'BBOX_TopLeft_y', 'BBOX_BottomRight_x', 'BBOX_BottomRight_y']].interpolate(method='linear', axis=0)   
                df_new_tracker[['BBOX_TopLeft_x', 'BBOX_TopLeft_y', 'BBOX_BottomRight_x', 'BBOX_BottomRight_y']] = bbox_position
                interpolated_df_final_list.append(df_new_tracker)

        interpolated_df = pd.concat(interpolated_df_final_list, ignore_index=True)
        missing_vidTimer_df = self.find_missing_vidTimer(dataframe, vidTimer_present_in_interpolated_df)
        interpolated_final_df = pd.concat([interpolated_df, missing_vidTimer_df], ignore_index=True).sort_values(by=['Video_Internal_Timer']).reset_index(drop=True)
        return interpolated_final_df

    def velocity_estimation(self, interpolated_df, rolling_window_size=15, video_fps=30):
        """Groups the entire .csv by tracker_id and rolling average on each of the BBOX Coordinates to remove noise from detections. 
        Then calculating the center point of the base of each bbox and finding their correspoding world coordinates using homography matrix
        Calculating the distance between points in consecutive frames and converting them into km/h.

        Args:
            interpolated_df (pd.Dataframe): Dataframe after interpolation
            rolling_window_size (int, optional): Samples to consider for rolling average. Defaults to 15.
            video_fps (int, optional): Data sample rate of the dataframe. Defaults to 30.

        Returns:
            df_interpolated_dup (pd.Dataframe): Dataframe containing the speed of each tracker in a separate column 'Speed'
        """
        camera_calib = Calibration()
        df_interpolated_dup = interpolated_df.copy()
        unique_trackers = df_interpolated_dup.Tracker_ID.unique()
        tracker_group = df_interpolated_dup.groupby('Tracker_ID')
        final_tracker_list = []
        for unique_tracker_id in unique_trackers: # Looping for each unique Tracker_ID
            if not pd.isna(unique_tracker_id):
                single_tracker_group = tracker_group.get_group(unique_tracker_id)
                bbox_positions = single_tracker_group[['Video_Internal_Timer', 'BBOX_TopLeft_x', 'BBOX_TopLeft_y', 'BBOX_BottomRight_x', 'BBOX_BottomRight_y', 'Class_ID']]
                bbox_positions['BBOX_TopLeft_x'] = bbox_positions['BBOX_TopLeft_x'].rolling(window=rolling_window_size).mean()
                bbox_positions['BBOX_TopLeft_y'] = bbox_positions['BBOX_TopLeft_y'].rolling(window=rolling_window_size).mean()
                bbox_positions['BBOX_BottomRight_x'] = bbox_positions['BBOX_BottomRight_x'].rolling(window=rolling_window_size).mean()
                bbox_positions['BBOX_BottomRight_y'] = bbox_positions['BBOX_BottomRight_y'].rolling(window=rolling_window_size).mean()
                

                bbox_positions = list(bbox_positions.to_records(index=False))
                prev_point = -1
                velocity_estimation = []

                for vid_timer, x1, y1, x2, y2, class_id in bbox_positions: 
                    center_x = (x1 + x2)/2
                    if class_id in (0,1,2): 
                        _, max_y = sorted((y1, y2))
                    elif class_id in (3,4,5,6):
                        max_y = (y1 + y2)/2
                    base_coordinate = camera_calib.projection_pixel_to_world((center_x, max_y)) # Calculating the center of point of the bottom edge of BBOX and calculating it's world coordinates with homography
                    current_point = (center_x, max_y)

                    if prev_point == -1:
                        new_row_withSpeed = {
                            'Video_Internal_Timer': vid_timer, 'Speed': 0, 'Arrow_points': list([0,0])
                        }   
                    else:
                        distance_metres = float(math.sqrt(math.pow(prev_point[0] - base_coordinate[0], 2) + math.pow(prev_point[1] - base_coordinate[1], 2))) # Finding the euclidean distance between current and previous point
                        speed_kmH = float(distance_metres * self.video_fps * 3.6) # Converting meters/s to km/h and update rate is equal to video's fps (default 30, this is also the rate at which data is sampled in the .csv files)
                        previous_point = previous_point

                        if speed_kmH < 1:
                            speed_kmH = 0 # To prevent small movements in the BBOX_Positions when the VRU's are standing 

                        new_row_withSpeed = {
                            'Video_Internal_Timer': vid_timer, 'Speed': speed_kmH, 'Arrow_points': [previous_point[0], previous_point[1]]
                        }
                    velocity_estimation.append(new_row_withSpeed)
                    prev_point = base_coordinate
                    previous_point = current_point

                    
                ve_df = pd.DataFrame(velocity_estimation)
                ve_df['Speed'] = ve_df['Speed'].round(1)
                
                
                # Inserting the 'Speed' Values back into the tracker group
                single_tracker_group['Speed'] = single_tracker_group['Video_Internal_Timer'].map(ve_df.set_index('Video_Internal_Timer')['Speed'])
                single_tracker_group['Arrow_points'] = single_tracker_group['Video_Internal_Timer'].map(ve_df.set_index('Video_Internal_Timer')['Arrow_points'])
                final_tracker_list.append(single_tracker_group)

        # Inserting all the modified tracker values back into the originial dataframe
        final_ve_tracker_df = pd.concat(final_tracker_list)
        df_interpolated_dup = df_interpolated_dup.join(final_ve_tracker_df['Speed'])
        df_interpolated_dup = df_interpolated_dup.join(final_ve_tracker_df['Arrow_points'])
        return df_interpolated_dup

    def conf_score_based_class_id_matching(self, df_speed):
        df_speed_dup = df_speed.copy()
        unique_trk = df_speed_dup.Tracker_ID.unique()
        tracker_group = df_speed_dup.groupby('Tracker_ID')
        cleanlist_trackers = [x for x in unique_trk if str(x) != 'nan']
        for unique_trk_id in cleanlist_trackers:
            single_trk_group = tracker_group.get_group(unique_trk_id)
            
            # group of different class_ids occuring for the same tracker_id
            class_ids_in_single_traker = single_trk_group.Class_ID.unique()
            class_id_grp = single_trk_group.groupby('Class_ID')

            sum_dict = {}
            for class_id in class_ids_in_single_traker:
                if str(class_id)!='nan':
                    single_class_grp = class_id_grp.get_group(class_id)
                    sum = single_class_grp.Conf_Score.sum()
                    sum_dict[class_id]= sum

            for key,v in sum_dict.items():
                if v == max(sum_dict.values()):
                    corrected_class_id = key
            
            # correct the class_id based on maximum confidence_theshold among certain classes for a single tracker_id.
            df_speed_dup.loc[df_speed_dup['Tracker_ID']==unique_trk_id, 'Class_ID'] = corrected_class_id
        
        return df_speed_dup

    def class_id_matching(self, df):
        """
        Eliminates switching class_id and allocates the one class_id for each tracker
        Processing steps:
        1. Groups the entire dataframe by unique tracker id and calculates the standard deviation of Class_ID for each tracker
        2. If standard deviation is less than 0.35, we are simply allocating the most frequently occurring Class_ID (calculated using pd.Mode()) to the entire tracker
        3. If standard deviation is more than 0.35
            3.1 Considering only non-zero speeds for each tracker (Considering speeds only when they are actually moving)
            3.2 Ignoring rows of data if the bbox_coordinates lie within the specified ignorance regions
            3.3 If mean speed of the tracker is more than 9 km/h, we are excluding the class_id 1 (pedestrian) and assigning the tracker with the most common occurring between escooter and cyclist
            3.4 If 95-percentile of max speed of the tracker is equal to or more than 23 km/h, we are assigning simply class_id 2 (cyclist)
            3.5 If both of the above conditions are not met, we are simply allocating the most frequently occurring Class_ID to the entire tracker (just like in step 2 but now with considerations of 3.1 and 3.2)

        Args:
            df (pd.Dataframe): Dataframe with speed data for each tracker

        Returns:
            final_df (pd.Dataframe): Dataframe with corrected class_id_matching
        """
        df_speed = df.copy()
        df_speed_dup = df.copy()
        # Consider only Escooter, Pedestrian, Cyclist classes and empty frames
        df_speed_dup = df_speed_dup.loc[pd.isna(df_speed_dup['Class_ID']) | df_speed_dup['Class_ID']<3]
        unique_trackers = df_speed_dup.Tracker_ID.unique()
        tracker_group = df_speed_dup.groupby('Tracker_ID')

        cleanlist_trackers = [x for x in unique_trackers if str(x) != 'nan']
        for unique_tracker_id in cleanlist_trackers:
            single_tracker_group = tracker_group.get_group(unique_tracker_id)
            
            std_class_ID = df_speed_dup.loc[df_speed_dup['Tracker_ID'] == unique_tracker_id, 'Class_ID'].std() 
            freq_occuring_class_id = df_speed_dup.loc[df_speed_dup['Tracker_ID'] == unique_tracker_id, 'Class_ID'].mode()[0]

            if std_class_ID < 0.35:
                # Allocating the most frequently occurring class_id
                df_speed_dup.loc[df_speed_dup['Tracker_ID'] == unique_tracker_id, 'Class_ID'] = freq_occuring_class_id

            else:
                df_speed_second_dup = df_speed_dup.copy()
                df_speed_second_dup.loc[((df_speed_second_dup['Tracker_ID'] == unique_tracker_id) & (df_speed_second_dup['Speed'] == 0)), ['Class_ID', 'Speed']] = np.nan

                # Ignorance Regions - BBOX Coordinates within these regions are ignored
                top_left_corner = np.array([[495,118], [590,173], [625,221], [673,225], [641,162], [680,145], [588,92]])
                right_section = np.array([[1373,402], [1379,461], [1497,471], [1489,411]])
                
                for index, row in single_tracker_group.iterrows():
                    x1, y1, x2, y2 = row['BBOX_TopLeft_x'], row['BBOX_TopLeft_y'], row['BBOX_BottomRight_x'], row['BBOX_BottomRight_y']
                    center_x = (x1+x2)/2
                    if freq_occuring_class_id in [0, 1, 2]:
                        _, y = sorted((y1, y2))
                    else:
                        y = (y1+y2)/2

                    is_bbox_present_right = cv2.pointPolygonTest(right_section, (center_x, y), False)
                    is_bbox_present_Topleft = cv2.pointPolygonTest(top_left_corner, (center_x, y), False)
                    if (is_bbox_present_right == 1 or is_bbox_present_Topleft == 1): # Ignoring bbox coordinates present inside the ignorance regions
                        df_speed_second_dup.iloc[index, [df_speed_second_dup.columns.get_loc(c) for c in ['Class_ID', 'Speed']]] = np.nan

                corrected_speed_mean = df_speed_second_dup.loc[df_speed_second_dup['Tracker_ID'] == unique_tracker_id, 'Speed'].mean()
                corrected_speed_95percentile = df_speed_second_dup.loc[df_speed_second_dup['Tracker_ID'] == unique_tracker_id, 'Speed'].quantile(0.95)
                
                if freq_occuring_class_id in [0, 1, 2]:
                    if corrected_speed_95percentile >= 23:
                        df_speed_dup.loc[df_speed_dup['Tracker_ID'] == unique_tracker_id, 'Class_ID'] = 2

                    elif corrected_speed_mean > 9:
                        keys = df_speed_second_dup.loc[df_speed_second_dup['Tracker_ID'] == unique_tracker_id, 'Class_ID'].value_counts().keys().tolist()
                        counts = df_speed_second_dup.loc[df_speed_second_dup['Tracker_ID'] == unique_tracker_id, 'Class_ID'].value_counts().tolist()
                        for freq in zip(keys, counts):# Ignoring Class_ID = 1 (pedestrian) and assigning the first value (since the counts is already in descending order, the first elements are obviously the more frequently occurring ;)
                            if keys != 1: 
                                df_speed_dup.loc[df_speed_dup['Tracker_ID'] == unique_tracker_id, 'Class_ID'] = freq[0]
                                break
                
                else:
                    # If trackers are empty after removing ignorance regions, then use before-ignorance-region tracker data to get the most frequently occurring Class_ID
                    df_speed_dup = self.conf_score_based_class_id_matching(df_speed_dup)

        # Copy only changed indexes to the original dataframe
        for i in df_speed_dup.index:
            df_speed.loc[i, :] = df_speed_dup.loc[i, :]
        
        return df_speed
        
    def run(self): # Main function of the class which runs all the post-processing and saves the video
        df_duplicate = self.detections_dataframe.copy()
        removed_df = self.remove_tracker(df_duplicate)
        print('\n-> Finished cleaning trackers')
        removed_df.to_csv(f'{self.output_directory}/{self.file_name}_cleaned.csv')

        interpolated_df = self.interpolate_data(removed_df)
        interpolated_df.to_csv(f'{self.output_directory}/{self.file_name}_interpolated.csv')
        print('-> Finished interpolating missing tracker coordinates')

        speed_df = self.velocity_estimation(interpolated_df)
        speed_df.to_csv(f'{self.output_directory}/{self.file_name}_speed.csv')
        print('-> Finished calculating the velocities')

        self.final_df = self.class_id_matching(speed_df)
        self.final_df.to_csv(f'{self.output_directory}/{self.file_name}_final.csv')
        print('-> Finished Class_ID Matching')
        # print('\nNow, saving the video ...')

        df_with_index = self.group_by_internalTimer_with_index(self.final_df)
        df_latest = self.Save_angle_to_csv(df_with_index, self.final_df)
        df_latest.to_csv(f'{self.output_directory}/{self.file_name}_latest.csv')
        print('-> Finished Saving Heading_angle')
        print('\nNow, saving the video ...')

        # Save video
        self.groupedByFrametime = self.group_by_internalTimer(self.final_df)
        self.multi_process()

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default=None, help=['path to .csv data'])
    parser.add_argument('--input_video', type=str, default=None, help=['input video file, should be the same one used in .csv'])
    parser.add_argument('--output_video', type=str, default=None, help=['path to save result(s)'])
    parser.add_argument('--enable_minimap', default=False, action='store_true', help='provied option for showing the minimap in result -- True (or) False')
    parser.add_argument('--enable_trj_mode', default=False, action='store_true', help='provied option to turn on or off the trjectory recording -- True (or) False')
    parser.add_argument('--trajectory_update_rate', type=int, default=30, help='provide a number to update a trajectory after certain frames')
    parser.add_argument('--save_class_frames', type=int, default=0, help='Save frames of requied class from 0 to 6 classes\
                                                    (0-Escooter, 1-Pedestrian, 2-Cyclist, 3-Motorcycle, 4-Car, 5-Truck, 6-Bus)')    
    opt = parser.parse_args()
    print("---- Traffic Camera Tracking (CARISSMA) ----")
    print("---- Post-Processing ----")
    return opt

if __name__ == '__main__':
    opt = parser_opt()
    obj = PostProcess(**vars(opt))
    obj.run()
