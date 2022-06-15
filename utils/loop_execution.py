import os
import pathlib

video_path = pathlib.Path('/home/mobilitylabextreme002/Desktop/video_capture')
# start_time = 17
# end_time = 18

print(f'Total number of videos: {len(list(video_path.iterdir()))}')

filtered_videos_count = 0
for videos in video_path.iterdir():
    # hour_timestamp = int(videos.stem.split('-')[-1][:2])
    # if hour_timestamp >= start_time and hour_timestamp <= end_time:
    filtered_videos_count += 1
    #print(videos)
    command = f'python inference.py --input {videos} --model_weights yolo_v5_main_files/runs/train/new_classes_filtered/weights/best.engine --output /home/mobilitylabextreme002/Videos/outputs/loop_execution_all/{videos.stem}.mkv --minimap --trj_mode'
    os.system(command)

    # if filtered_videos_count >= 20:
    #     print('Finished successfully !!')
    #     exit()
# print(f'Filtered number of videos: {filtered_videos_count}')