import numpy as np, matplotlib.pyplot as plt, pandas as pd
import matplotlib.cm as cm
import matplotlib.patches as patch
import cv2

def detect_tracker_disjoint(trk_group):
    # Check whether a tracker is disjoint or not (If a tracker enters and leaves the frame in defined points, it is a perfect tracker (return 0). Otherwise, it is disjoint (return 1))

    def _check_within_bounds(point, boundary_regions):
        is_within_bounds = False
        for boundary_region in boundary_regions:
            if point[0] >= boundary_region[0][0] and point[0] <= boundary_region[1][0] and point[1] >= boundary_region[0][1] and point[1] <= boundary_region[1][1]:
                is_within_bounds = True
                return (is_within_bounds, boundary_region)
        return (is_within_bounds, None)
    
    entry_exit_regions = [
        ((0, 0), (150, 200)),
        ((593, 0), (826, 91)),
        ((1290, 0), (1490, 90)),
        ((1550, 215), (1800, 640)),
        ((1247, 713), (1405, 842)),
        ((797, 938), (1250, 1080)),
        ((0, 720), (293, 1080)),
        ((0, 338), (142, 560))
    ]

    # x1, y1 = (trk_group['BBOX_TopLeft'].iloc[0])[1:-1].split(',')
    # x2, y2 = (trk_group['BBOX_BottomRight'].iloc[0])[1:-1].split(',')
    foot_point_initial = ((int(trk_group['BBOX_TopLeft_x'].iloc[0]) + int(trk_group['BBOX_BottomRight_x'].iloc[0]))/2, int(trk_group['BBOX_BottomRight_y'].iloc[0]))
    # x1, y1 = (trk_group['BBOX_TopLeft'].iloc[-1])[1:-1].split(',')
    # x2, y2 = (trk_group['BBOX_BottomRight'].iloc[-1])[1:-1].split(',')
    foot_point_final = ((int(trk_group['BBOX_TopLeft_x'].iloc[-1]) + int(trk_group['BBOX_BottomRight_x'].iloc[-1]))/2, int(trk_group['BBOX_BottomRight_y'].iloc[-1]))
    # _foot_point_final = ((int(x1) + int(x2))/2, int(y2))
    
    initial_bound_check, _boundary_region_initial = _check_within_bounds(foot_point_initial, entry_exit_regions)
    final_bound_check, _boundary_region_final = _check_within_bounds(foot_point_final, entry_exit_regions)

    if (initial_bound_check and final_bound_check) and (_boundary_region_final is not _boundary_region_initial):
        return 0
    else:
        return 1

def create_tracker_search_regions(single_tracker_group):
    """Creates a decagon search region (opencv contour) around the tracker to check

    Args:
        single_tracker_group (pd.df): Dataframe containing one single tracker id

    Returns:
        np.array: Search region in opencv2 contours format
    """
    min_search_side_length=50
    max_search_side_length=120

    min_magnitude = 0
    max_magnitude = 20

    slope = (max_search_side_length - min_search_side_length) / (max_magnitude - min_magnitude)
    _tracker_association_buffer_point = 10
    
    last_point_coordinate = np.array([single_tracker_group['Minimap_x'].iloc[-1], single_tracker_group['Minimap_y'].iloc[-1]])
    
    try:    
        buffer_point_coordinate = np.array([single_tracker_group['Minimap_x'].iloc[-_tracker_association_buffer_point], single_tracker_group['Minimap_y'].iloc[-_tracker_association_buffer_point]])
    except IndexError:
        _tracker_association_buffer_point = 2
        buffer_point_coordinate = np.array([single_tracker_group['Minimap_x'].iloc[-_tracker_association_buffer_point], single_tracker_group['Minimap_y'].iloc[-_tracker_association_buffer_point]])
    if single_tracker_group.shape[0] > _tracker_association_buffer_point:
        directional_vector = last_point_coordinate - buffer_point_coordinate
        magnitude = np.linalg.norm(directional_vector)
        search_side_length = slope * magnitude + min_search_side_length

    search_region = []

    # Creating a decagon (circle will have too many points, so approximating it to a 10-point polygon)
    for i in range(10):
        x = last_point_coordinate[0] + search_side_length * np.cos(i*2*np.pi/10)
        y = last_point_coordinate[1] + search_side_length * np.sin(i*2*np.pi/10)
        search_region.append((x, y))

    return np.array(search_region, dtype=np.float32).reshape((-1, 1, 2))

def combine_trackers(orig_dataframe, first_tracker_group, second_tracker_group):
    """Combines two tracker group dataframes

    Args:
        orig_dataframe (pd.df): Original pandas dataframe
        first_tracker_group (pd.df): First tracker group
        second_tracker_group (pd.df): Second tracker group
    Returns:
        (pd.df): Combined first and second tracker group using interpolation (Tracker ID will be the same one as the first tracker group)
    """
    video_timer_df = orig_dataframe['Video_Internal_Timer'].unique()
    first_tracker_id = first_tracker_group.iloc[0]['Tracker_ID']
    second_tracker_id = second_tracker_group.iloc[0]['Tracker_ID']
    new_tracker_combined_df = pd.concat([first_tracker_group, second_tracker_group],  ignore_index=True)
    
    new_tracker_combined_dict = new_tracker_combined_df.to_dict('records')
    min_vid_timer = first_tracker_group['Video_Internal_Timer'].max()
    max_vid_timer = second_tracker_group['Video_Internal_Timer'].min()
    
    for index in range(np.where(video_timer_df == min_vid_timer)[0][0], np.where(video_timer_df == max_vid_timer)[0][0] + 1):
        if video_timer_df[index] not in new_tracker_combined_df['Video_Internal_Timer'].values:
            # If Tracker_ID is not present in this specific Video_Internal_Timer, creating a new row by having default values as None
            # These missing Tracker_ID are later interpolated
            idx = orig_dataframe.index[orig_dataframe['Video_Internal_Timer'] == video_timer_df[index]][0]
            row_tracker_id = orig_dataframe.iloc[idx]
            new_row = {
                'Video_Internal_Timer': video_timer_df[index], 
                'Date': row_tracker_id['Date'], 'Time': row_tracker_id['Time'], 'Millisec': row_tracker_id['Millisec'], 'Tracker_ID': first_tracker_id, 
                'Class_ID': np.nan, 'Conf_Score': np.nan, 'BBOX_TopLeft_x': np.nan, 'BBOX_TopLeft_y': np.nan,
                'BBOX_BottomRight_x': np.nan, 'BBOX_BottomRight_y': np.nan, 'BBOX_x_foot': np.nan, 'Minimap_x': np.nan, 'Minimap_y': np.nan
            }
            new_tracker_combined_dict.append(new_row)
        
    # Converting the list of rows to Dataframe and interpolating the missing BBOX_Values.
    df_new_tracker = pd.DataFrame(new_tracker_combined_dict)
    df_new_tracker = df_new_tracker.assign(Tracker_ID=first_tracker_id)
    df_new_tracker = df_new_tracker.sort_values(by=['Video_Internal_Timer']).reset_index(drop=True)
    # try:
    #     # Using Spline Interpolation by default.
    #     bbox_position = df_new_tracker[['BBOX_TopLeft_x', 'BBOX_TopLeft_y', 'BBOX_BottomRight_x', 'BBOX_BottomRight_y', 'BBOX_x_foot', 'Minimap_x', 'Minimap_y']].interpolate(method='spline', order=4, axis=0)
    # except ValueError:
    #     # If spline interpolation fails, falls back to simple linear interpolation
    bbox_position = df_new_tracker[['BBOX_TopLeft_x', 'BBOX_TopLeft_y', 'BBOX_BottomRight_x', 'BBOX_BottomRight_y', 'BBOX_x_foot', 'Minimap_x', 'Minimap_y']].interpolate(method='linear', axis=0)   
    df_new_tracker[['BBOX_TopLeft_x', 'BBOX_TopLeft_y', 'BBOX_BottomRight_x', 'BBOX_BottomRight_y', 'BBOX_x_foot', 'Minimap_x', 'Minimap_y']] = bbox_position.rolling(window=15, min_periods=1).mean()
    
#     fig, ax = plt.subplots(nrows=1, ncols=3)
#     fig.set_figheight(5)
#     fig.set_figwidth(10)
    
#     first_tracker_group.plot(x='Minimap_x', y='Minimap_y', ax=ax[0], title=f"{first_tracker_id}")
#     second_tracker_group.plot(x='Minimap_x', y='Minimap_y', ax=ax[1], title=f"{second_tracker_id}")
#     df_new_tracker.plot(x='Minimap_x', y='Minimap_y', ax=ax[2], title=f"Combined")

#     plt.setp(ax, xlim=(0, 400), ylim=(0, 400))
#     fig.savefig(f'..\\tracker_association\\{first_tracker_id}_{second_tracker_id}.png', facecolor='white', transparent=False)
#     df_new_tracker.to_csv(f'..\\tracker_association\\{first_tracker_id}_{second_tracker_id}.csv')
   
    return df_new_tracker

def tracker_fusion(main_df, tracker_search_time_threshold = 30):
    """Combines all disjoint tracker of single VRU into one tracker using spatial search regions. Only for VRU's (Class_ID < 3).

    Args:
        main_df (pd.df): interpolated dataframe
        tracker_search_time_threshold (int, optional): Number of seconds until which a new matching tracker is searched for. Defaults to 30.
    Returns:
        df_final_fused (pd.df): Tracker fused dataframe
        (total_num_trackers, num_disjoint_trackers, num_prev_disjoint_perfected, num_perfect_trackers): Some statistics for debugging
    """
    tracker_group = main_df.groupby('Tracker_ID')
    unique_trackers = main_df['Tracker_ID'].unique()

    counter = 0

    tracker_list = [tracker_group.get_group(unique_tracker_id).reset_index(drop=True).dropna() for unique_tracker_id in unique_trackers if not pd.isna(unique_tracker_id)]
    tracker_list = [trk_group for trk_group in tracker_list if not trk_group.empty]
    tracker_finished_list = []

    num_perfect_trackers = 0
    num_disjoint_trackers = 0
    num_prev_disjoint_perfected = 0
    total_num_trackers = len(tracker_list)

    while len(tracker_list) > 0:  
        counter += 1
        main_tracker_group = tracker_list[0]
        main_tracker_final_time = main_tracker_group.iloc[-1]['Time']
        main_tracker_classID = main_tracker_group['Class_ID'].mode()[0]
        is_disjoint = detect_tracker_disjoint(main_tracker_group)
        
        if is_disjoint and main_tracker_classID < 3: # Only for VRU's. Vehicles have class_id > 3
            search_region = create_tracker_search_regions(main_tracker_group)
            store_neighborhood_trackers = []
            for loop_tracker_index in range(1, len(tracker_list)): # Looping through all following tracker
                loop_tracker_group = tracker_list[loop_tracker_index] 
                loop_tracker_initial_time = loop_tracker_group.iloc[0]['Time']
                _time_delta = (loop_tracker_initial_time - main_tracker_final_time).seconds

                if _time_delta <= tracker_search_time_threshold: # If time delta between looped tracker is within threshold of main tracker, check whether they are close enough
                    loop_tracker_foot_point_initial = (int(loop_tracker_group['Minimap_x'].iloc[0]), int(loop_tracker_group['Minimap_y'].iloc[0]))    
                    does_loop_tracker_lie_in_neighborhood = cv2.pointPolygonTest(search_region, loop_tracker_foot_point_initial, False)
                    if does_loop_tracker_lie_in_neighborhood >= 1:
                        store_neighborhood_trackers.append((loop_tracker_index, _time_delta))

            if len(store_neighborhood_trackers) > 0:
                closest_neighbor_tracker_index = None
                store_neighborhood_trackers = np.array(store_neighborhood_trackers)
                store_neighborhood_trackers = np.sort(store_neighborhood_trackers, axis=0) # Getting the closest neighbor based on time

                closest_neighbor_tracker_index = store_neighborhood_trackers[0, 0]
                closed_neighbor_tracker_group = tracker_list[closest_neighbor_tracker_index]
                new_tracker = combine_trackers(main_df, main_tracker_group, closed_neighbor_tracker_group) # Combines the main_tracker and closest_neighbor

                # Deleting the associated tracker and inserting the new combine one
                num_prev_disjoint_perfected += 1
                tracker_list.pop(closest_neighbor_tracker_index)
                tracker_list.pop(0)
                tracker_list.insert(0, new_tracker)

            else:
                # If no neighboring trackers found, add to the finished the list
                tracker_list.pop(0)
                num_disjoint_trackers += 1
                tracker_finished_list.append(main_tracker_group)
        else:
            # If the tracker is perfect or Class_ID > 3 (vehicles), add directly to the finished list
            tracker_list.pop(0)
            num_perfect_trackers += 1
            tracker_finished_list.append(main_tracker_group)

    # Renumbering Tracker_ID's
    MIN_COUNT_NUMBER = 1
    MAX_COUNT_NUMBER = len(tracker_finished_list) + 1

    tracker_final_list = []
    for final_num, trk in zip(range(MIN_COUNT_NUMBER, MAX_COUNT_NUMBER), tracker_finished_list):
        trk = trk.assign(Tracker_ID=final_num)
        tracker_final_list.append(trk)
    
    # Combining everything into a Pandas Dataframe
    df_final_fused = pd.concat(tracker_final_list, ignore_index=True)
    df_final_fused = df_final_fused.sort_values(by=['Video_Internal_Timer']).reset_index(drop=True)
    
    return df_final_fused, (total_num_trackers, num_disjoint_trackers, num_prev_disjoint_perfected, num_perfect_trackers)