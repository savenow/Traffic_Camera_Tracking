import os
import pathlib

video_path = pathlib.Path('/home/mobilitylabextreme002/Desktop/video_capture')

start_date = 12
end_date = 14

start_time = 00
end_time = 23

print(f'Total number of videos: {len(list(video_path.iterdir()))}')

filtered_videos_count = 0
for videos in video_path.iterdir():
    day_timestamp = int(videos.stem.split('-')[0][:2])
    month_timstamp = int(videos.stem.split('-')[0][3:5])
    hour_timestamp = int(videos.stem.split('-')[-1][:2])
    if (hour_timestamp >= start_time and hour_timestamp <= end_time) and (day_timestamp >= start_date and day_timestamp <= end_date):
        filtered_videos_count += 1
        #print(videos)
        command = f'python inference.py --input {videos} --model_weights yolo_v5_main_files/runs/train/new_classes_filtered/weights/best.engine --output /home/mobilitylabextreme002/Videos/outputs/loop_execution_all/{videos.stem}_allClasses.mkv --minimap --trj_mode'
        os.system(command)

        # if filtered_videos_count >= 20:
        #     print('Finished successfully !!')
        #     exit()
print(f'Number of videos inferenced: {filtered_videos_count}')