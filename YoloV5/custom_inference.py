import torch
import time
import cv2
from yolo_v5_main_files.utils.torch_utils import time_sync

def video_inference(input_path, output_path, model_weights):
    video_capture = cv2.VideoCapture(input_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = total_frames / fps
    codec = 'mp4v'

    video_output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), float(fps), (width, height),)
    model = torch.hub.load(
        r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\YoloV5\yolo_v5_main_files', 
        'custom', 
        path=model_weights, 
        source='local'
    ) 

    avg_inference = 0
    avg_inference_fps = 0

    avg_video_write = 0
    avg_video_write_fps = 0
    framecount = 1
    t0 = time_sync()
    while video_capture.isOpened():
        _, frame = video_capture.read()

        if _:
            t1 = time_sync()
            results = model(frame)
            t2 = time_sync()
            inference_time = t2 - t1
            inference_fps  = 1/inference_time
            avg_inference += inference_time
            avg_inference_fps += inference_fps

            t3 = time_sync()
            video_output.write(results.render()[0])
            t4 = time_sync()
            video_write_time = t4 - t3
            video_write_fps = 1/video_write_time
            avg_video_write += video_write_time
            avg_video_write_fps += video_write_fps

            print(f'Processed Frame: {framecount} => Inference Time: ({inference_time:.3f}s) ({inference_fps:.3f} fps); Video File write time: ({video_write_time:.3f}s) ({video_write_fps:.3f} fps)')

            framecount += 1
        else:
            break
    
    if framecount > 1:
        avg_inference /= framecount
        avg_inference_fps /= framecount
        avg_video_write /= framecount
        avg_video_write_fps /= framecount

        print(f"\nFinished processing video. REPORT:\nTotal frames: {framecount}; Input Video Duration: {video_duration}; Total time required: {time_sync() - t0:.3f}")
        print(f'Average Inference Time per frame: {avg_inference}s ({avg_inference_fps} fps)')
        print(f'Average Video Write Time per frame: {avg_video_write}s ({avg_video_write_fps} fps)')
    
    video_capture.release()
    video_output.release()

# Model
# model = torch.hub.load(r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\YoloV5\yolo_v5_main_files', 'custom', path=r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\YoloV5\yolo_v5_main_files\runs\train\exp10\weights\best.pt', source='local')  # or yolov5m, yolov5l, yolov5x, custom
model_weight = r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Traffic_Camera_Tracking\YoloV5\yolo_v5_main_files\runs\train\exp10\weights\best.pt'
# Images
# img = r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new\31\images\frame_000154.png'  # or file, Path, PIL, OpenCV, numpy, list

# # Inference
# results = model(img)

# # Results
# img_output = results.render()[0]  # or .show(), .save(), .crop(), .pandas(), etc
# #file_open = cv2.imread(img_output)
# cv2.imshow('Windows', img_output[..., ::-1].copy())
# cv2.waitKey(0)

input = r'C:\Vishal-Videos\Project_Escooter_Tracking\samples\08-06-2021_10-40.mkv'
output = r'C:\Users\balaji\Desktop\Traffic_Camera_Tracking\Main_Code\Infered_Videos\Yolo\08-06-2021_10-40.mkv'
video_inference(input, output, model_weight)