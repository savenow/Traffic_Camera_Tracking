import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = r'C:\Vishal-Videos\Project_Escooter_Tracking\input_new\38\images\frame_000035.png'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.save()  # or .show(), .save(), .crop(), .pandas(), etc.