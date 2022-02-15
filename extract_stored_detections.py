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
warnings.filterwarnings("ignore") # To ignore certain warnings from Pandas

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
        # At the end of .csv file, some erroneous data from tracker has been found with Video_Internal_Timer = 0. The following code snippet would remove them from the dataframe
        last_row_df = self.detections_dataframe.shape[0] - 1
        if self.detections_dataframe.iloc[last_row_df]['Video_Internal_Timer'] == 0:
            last_index_drop = last_row_df + 1
            df_rev = self.detections_dataframe[::-1]
            start_index_drop = -1    
            for row, col in df_rev.iterrows():
                if col['Video_Internal_Timer'] != 0:
                    start_index_drop = row + 1
                    break
            self.detections_dataframe.drop(self.detections_dataframe.index[start_index_drop:last_index_drop], 0, inplace=True)
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
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                vid_timer = int(self.video_cap.get(cv2.CAP_PROP_POS_MSEC))
                for data in list_grouped_by_frametimes:
                    df_frametime = data[0]['Video_Internal_Timer']
                    
                    # Checking for interal_timer from .csv file and matching it with the internal timer from video file (For syncing frames)
                    if df_frametime == vid_timer:
                        framecounter += 1
                        print(f"[INFO] Processing frame {framecounter}")
                        
                        # Timestamp-Text properties
                        font = cv2.FONT_HERSHEY_SIMPLEX
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

        print(f'[INFO] Finished saving video file.Total Number of frames: {framecounter}')

        self.video_writer.release()
        self.video_cap.release()

    def run(self):
        frametime_group_list = self.group_by_internalTimer()
        self.groupedData_toVideoWriter(frametime_group_list)

class PostProcess():
    def __init__(self, data_file, input_video, output_video, enable_minimap, enable_trj_mode):
        self.detections_dataframe = pd.read_csv(data_file, index_col=[0])
        self.file_name = Path(output_video).stem
        
        # At the end of .csv file, some erroneous data from tracker has been found with Video_Internal_Timer = 0. The following code snippet would remove them from the dataframe
        last_row_df = self.detections_dataframe.shape[0] - 1
        if self.detections_dataframe.iloc[last_row_df]['Video_Internal_Timer'] == 0:
            last_index_drop = last_row_df + 1
            df_rev = self.detections_dataframe[::-1]
            start_index_drop = -1    
            for row, col in df_rev.iterrows():
                if col['Video_Internal_Timer'] != 0:
                    start_index_drop = row + 1
                    break
            self.detections_dataframe.drop(self.detections_dataframe.index[start_index_drop:last_index_drop], 0, inplace=True)
        
        self.video_cap = cv2.VideoCapture(input_video)
        frame_width = int(self.video_cap.get(3))
        frame_height = int(self.video_cap.get(4))
        self.video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))
        self.Visualize = Visualizer(enable_minimap, enable_trj_mode)

    def group_by_internalTimer(self, df):
        framecounter = 0
        prev_frametime = -1
        list_grouped_by_frametimes = []
        detection_per_frame = []

        for index, row in df.iterrows():
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
    
    def groupedData_toVideoWriter(self, df, list_grouped_by_frametimes):
        framecounter = 0
        min_vid_timer = int(df['Video_Internal_Timer'].min())
        max_vid_timer = int(df['Video_Internal_Timer'].max())
        
        self.video_cap.set(cv2.CAP_PROP_POS_MSEC, min_vid_timer)
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            if ret:
                vid_timer = int(self.video_cap.get(cv2.CAP_PROP_POS_MSEC))
                if vid_timer > max_vid_timer:
                    break
                for data in list_grouped_by_frametimes:
                    df_frametime = data[0]['Video_Internal_Timer']
                    
                    # Checking for interal_timer from .csv file and matching it with the internal timer from video file (For syncing frames)
                    if df_frametime == vid_timer:
                        framecounter += 1
                        print(f"[INFO] Saving frame {framecounter}")
                        
                        for detection in data:
                            if not pd.isna(detection['Speed']):
                                # Drawing Trackers
                                outer_array = []
                                detection_array = []
                                x1 = detection['BBOX_TopLeft_x']
                                y1 = detection['BBOX_TopLeft_y']
                                x2 = detection['BBOX_BottomRight_x']
                                y2 = detection['BBOX_BottomRight_y']
                                detection_array.append(int(x1))
                                detection_array.append(int(y1))
                                detection_array.append(int(x2))
                                detection_array.append(int(y2))
                                if not pd.isna(detection['Class_ID']):
                                    detection_array.append(detection['Conf_Score']/100)
                                    detection_array.append(detection['Class_ID'])
                                else:
                                    detection_array.append(-1)
                                    detection_array.append(0)
                                detection_array.extend([0, 0, 0]) # Placeholder values. The visualizer function doesn't need these but kept in places to align with the indices.
                                detection_array.append(detection['Tracker_ID'])
                                detection_array.append(detection['Speed'])
                                outer_array.append(detection_array)
                                image = self.Visualize.drawAll(outer_array, frame, framecounter)

                            elif not pd.isna(detection['Tracker_ID']):
                                # Drawing Trackers
                                outer_array = []
                                detection_array = []
                                x1 = detection['BBOX_TopLeft_x']
                                y1 = detection['BBOX_TopLeft_y']
                                x2 = detection['BBOX_BottomRight_x']
                                y2 = detection['BBOX_BottomRight_y']
                                detection_array.append(int(x1))
                                detection_array.append(int(y1))
                                detection_array.append(int(x2))
                                detection_array.append(int(y2))
                                if not pd.isna(detection['Class_ID']):
                                    detection_array.append(detection['Conf_Score']/100)
                                    detection_array.append(detection['Class_ID'])
                                else:
                                    detection_array.append(-1)
                                    detection_array.append(0)
                                detection_array.extend([0, 0, 0]) # Placeholder values. The visualizer function doesn't need these but kept in places to align with the indices.
                                detection_array.append(detection['Tracker_ID'])
                                outer_array.append(detection_array)
                                image = self.Visualize.drawTracker(outer_array, frame, framecounter)
                                
                            elif not pd.isna(detection['Class_ID']):
                                # Drawing just BBOXes
                                outer_array = []
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
                                image = self.Visualize.drawBBOX(outer_array, frame, framecounter)
                            else:
                                # No Detections/Trackers. Just drawing the minimap (if enabled)
                                image = self.Visualize.drawEmpty(frame, framecounter)
                            
                        self.video_writer.write(image)
            else:
                break
        self.video_writer.release()        
    
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
                bbox_positions = single_tracker_group[['Video_Internal_Timer', 'BBOX_TopLeft_x', 'BBOX_TopLeft_y', 'BBOX_BottomRight_x', 'BBOX_BottomRight_y']]
                bbox_positions['BBOX_TopLeft_x'] = bbox_positions['BBOX_TopLeft_x'].rolling(window=rolling_window_size).mean()
                bbox_positions['BBOX_TopLeft_y'] = bbox_positions['BBOX_TopLeft_y'].rolling(window=rolling_window_size).mean()
                bbox_positions['BBOX_BottomRight_x'] = bbox_positions['BBOX_BottomRight_x'].rolling(window=rolling_window_size).mean()
                bbox_positions['BBOX_BottomRight_y'] = bbox_positions['BBOX_BottomRight_y'].rolling(window=rolling_window_size).mean()

                bbox_positions = list(bbox_positions.to_records(index=False))
                prev_point = -1
                velocity_estimation = []

                for vid_timer, x1, y1, x2, y2 in bbox_positions: 
                    center_x = (x1 + x2)/2 
                    _, max_y = sorted((y1, y2))
                    base_coordinate = camera_calib.projection_pixel_to_world((center_x, max_y)) # Calculating the center of point of the bottom edge of BBOX and calculating it's world coordinates with homography

                    if prev_point == -1:
                        new_row_withSpeed = {
                            'Video_Internal_Timer': vid_timer, 'Speed': 0
                        }   
                    else:
                        distance_metres = float(math.sqrt(math.pow(prev_point[0] - base_coordinate[0], 2) + math.pow(prev_point[1] - base_coordinate[1], 2))) # Finding the euclidean distance between current and previous point
                        speed_kmH = float(distance_metres * video_fps * 3.6) # Converting meters/s to km/h and update rate is equal to video's fps (default 30, this is also the rate at which data is sampled in the .csv files)
                        if speed_kmH < 1:
                            speed_kmH = 0 # To prevent small movements in the BBOX_Positions when the VRU's are standing 

                        new_row_withSpeed = {
                            'Video_Internal_Timer': vid_timer, 'Speed': speed_kmH
                        } 
                    velocity_estimation.append(new_row_withSpeed)
                    prev_point = base_coordinate
                    
                ve_df = pd.DataFrame(velocity_estimation)
                ve_df['Speed'] = ve_df['Speed'].round(1)
                
                # Inserting the 'Speed' Values back into the tracker group
                single_tracker_group['Speed'] = single_tracker_group['Video_Internal_Timer'].map(ve_df.set_index('Video_Internal_Timer')['Speed'])
                final_tracker_list.append(single_tracker_group)

        # Inserting all the modified tracker values back into the originial dataframe
        final_ve_tracker_df = pd.concat(final_tracker_list)
        df_interpolated_dup = df_interpolated_dup.join(final_ve_tracker_df['Speed'])
        return df_interpolated_dup

    def run(self):
        df_duplicate = self.detections_dataframe.copy()
        removed_df = self.remove_tracker(df_duplicate)
        removed_df.to_csv(f'{self.file_name}_cleaned.csv')

        interpolated_df = self.interpolate_data(removed_df)
        interpolated_df.to_csv(f'{self.file_name}_interpolated.csv')

        speed_df = self.velocity_estimation(interpolated_df)
        speed_df.to_csv(f'{self.file_name}_speed.csv')

        # Save video
        groupedByFrametime = self.group_by_internalTimer(speed_df)
        self.groupedData_toVideoWriter(speed_df, groupedByFrametime)

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default=None, help=['path to .csv data'])
    parser.add_argument('--input_video', type=str, default=None, help=['input video file, should be the same one used in .csv'])
    parser.add_argument('--output_video', type=str, default=None, help=['path to save result(s)'])
    parser.add_argument('--enable_minimap', default=False, action='store_true', help='provied option for showing the minimap in result -- True (or) False')
    parser.add_argument('--enable_trj_mode', default=False, action='store_true', help='provied option to turn on or off the trjectory recording -- True (or) False')
    opt = parser.parse_args()
    print("---- Traffic Camera Tracking (CARISSMA) ----")
    print("---- Post-Processing ----")
    return opt

if __name__ == '__main__':
    opt = parser_opt()
    # obj = ExtractFromCSV(**vars(opt))
    # obj.run()
    obj = PostProcess(**vars(opt))
    obj.run()
