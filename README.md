# Traffic Camera Tracking (SAVeNoW)

## Basic Overview
----
Main goals of this project is to **detect, track and obtain velocity information** of seven different traffic actors _(Escooter, Pedestrians, Cyclists, Motorcycle, Car, Truck, Bus)_ from a video footage of a traffic intersection in Ingolstadt.
This project is a part of **SaveNoW**, where all these data are used to create a digital twin of Ingolstadt for simulation purposes.

<img src="/readme_photos/Overview.png" width="800"> 


## Individual Components
----

### <u>Object Detection:</u>
_(Still actively developed..A lot of the parameters could change in future)_

PyTorch based [**YoloV5**](https://github.com/ultralytics/yolov5) framework is used for detection and the model *yolov5l6* is trained on custom dataset of around **15k** images (13k train/1.5k valid) in varied conditions (daylight, rain and night) using custom data, cityscapes and traffic data from COCO. All the images were manually labelled using CVAT online annotation tool. The model was trained by for 375 epochs using RTX 3090 at img size 1920 for multiple days to achieve final accuracy of 77%

For faster inference speeds, a _TensorRT Engine_ has been built from the traied weights and can infer at **125 fps** using RTX 3090

### <u>Tracker:</u>
Since YoloV5 is spatial CNN Object detector, it doesn't contain temporal information to recognize an object across multiple frames. For this purpose, a modified [**SORT (Simple Online Realtime Tracker)**](https://github.com/abewley/sort) has been used.


### <u>Postprocessing:</u>
The inference+tracker data is inherently noisy. So following postprocessing steps are applied to make it smooth and consistent:
1. Removing temporary noisy trackers
2. Interpolating missing tracker position
3. Estimating velocity for each tracker based on camera-image homography data and smoothening the trajectory with rolling average
4. ClassID matching to predict which class the tracker most probably belongs to. This also prevents classid switching of trackers
5. Visualize heading angle
6. Converting image coordinates to lat-long coordinates (again based on camera-image homography)
7. Finally visualizing it in a video

## Sample inference:
----

<img src="/readme_photos/sample_inference.gif" width="800"> 

## Usage:
`python inference.py --input {INPUT_VIDEO_PATH} --model_weights {WEIGHTS_PATH} --output {OUTPUT_PATH} --minimap --trj_mode`
## Notes
---- 
* Weights files and training datasets are not present in the repo. If you are interested, please feel free to contact us: pascal.brunner@carissma.eu
* Calibration values must be properly tuned depending on the individual scene (_Hint: Maybe will be automated in the future_)

## Contributing
----
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.