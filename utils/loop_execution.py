import os
import pathlib

video_path = pathlib.Path('/home/mobilitylabextreme002/Videos/fkk_new_videos/20220913_074500')
output_path = pathlib.Path('/home/mobilitylabextreme002/Videos/fkk_new_videos/20220913_074500_inference_output')


filtered_videos_count = 0
for videos in video_path.iterdir():
    input_videos = video_path / videos
    final_video_output_path = str(output_path / videos.stem) + '.mkv' 
    command = f'python inference.py --input {input_videos} --model_weights /home/mobilitylabextreme002/Desktop/weights/All_5_combined/weights/best.engine --output {final_video_output_path} --minimap --trj_mode'
    print(command)
    os.system(command)

# month = '06_2022'
# day_start = 2
# day_end = 9

# filtered_videos_count = 0
# for month_folders in video_path.iterdir():
#     if month_folders.stem == month:
#         month_folder_output_path = output_path / month
#         if not os.path.isdir(month_folder_output_path):
#             os.mkdir(month_folder_output_path)

#         month_folder_path = video_path / month
#         for day_folders in month_folder_path.iterdir():
#             num_day = int(str(day_folders.stem)[:2])
#             if num_day in [2, 3, 4, 5, 7, 8, 9]:
#                 #print(day_folders)
                
#                 day_folder_output_path = month_folder_output_path / f"{str(day_folders.stem)[:2]}_{month}"
#                 if not os.path.isdir(day_folder_output_path):
#                     os.mkdir(day_folder_output_path)
                
#                 for input_videos in day_folders.iterdir():
#                     final_video_output_path = str(day_folder_output_path / input_videos.stem) + ".mkv"
#                     command = f'python inference.py --input {input_videos} --model_weights yolo_v5_main_files/runs/train/new_classes_filtered/weights/best.engine --output {final_video_output_path} --minimap --trj_mode'
#                     #print(command)
#                     os.system(command)
        