import os
import pathlib

video_path = pathlib.Path('/home/mobilitylabextreme002/Desktop/video_capture')
output_path = pathlib.Path('/home/mobilitylabextreme002/Videos/outputs/loop_execution_2_6_to_9_6')

month = '06_2022'
day_start = 2
day_end = 9

filtered_videos_count = 0
for month_folders in video_path.iterdir():
    if month_folders.stem == month:
        month_folder_output_path = output_path / month
        if not os.path.isdir(month_folder_output_path):
            os.mkdir(month_folder_output_path)

        month_folder_path = video_path / month
        for day_folders in month_folder_path.iterdir():
            num_day = int(str(day_folders.stem)[:2])
            if num_day >= day_start and num_day <= day_end:
                #print(day_folders)
                
                day_folder_output_path = month_folder_output_path / f"{str(day_folders.stem)[:2]}_{month}"
                if not os.path.isdir(day_folder_output_path):
                    os.mkdir(day_folder_output_path)
                
                for input_videos in day_folders.iterdir():
                    final_video_output_path = str(day_folder_output_path / input_videos.stem) + ".mkv"
                    command = f'python inference.py --input {input_videos} --model_weights yolo_v5_main_files/runs/train/new_classes_filtered/weights/best.engine --output {final_video_output_path} --minimap --trj_mode'
                    #print(command)
                    os.system(command)
        