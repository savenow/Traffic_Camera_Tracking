# Traffic Camera Tracking (SAVeNoW)

## Basic Overview
----
Main goals of this project is to **detect, track and obtain velocity information** of three classes _(Escooter, Pedestrians and Cyclists)_ from a video footage of a traffic intersection in Ingolstadt.
This project is a part of **SaveNoW**, where all these data are used to create a digital twin of Ingolstadt for simulation purposes.

<img src="/readme_photos/Overview.png" width="600"> 

*For further updates on the project, follow this notion page: https://www.notion.so/Project-Traffic-Camera-tracking-15a9bb984c1341369ad40562610dc83f*


## Individual Components
----

### Object Detection:
_(Still actively developed..A lot of the parameters could change in future)_

PyTorch based [**YoloV5**](https://github.com/ultralytics/yolov5) framework is used for detection and the model *yolov5l6* is trained on custom dataset of around **89k** images in varied conditions (daylight, rain and night). All the images were manually labelled using CVAT online annotation tool. The model was trained by for 100+ hours using RTX 3090 for 150 epochs and reaches around 95% accuracy.

For faster inference speeds, a _TensorRT Engine_ has been built and can infer at **125 fps**


### Tracker:
Since YoloV5 is spatial CNN Object detector, it doesn't contain temporal information to recognize an object across multiple frames. For this purpose, a modified [**SORT (Simple Online Realtime Tracker)**](https://github.com/abewley/sort) has been used.

### Velocity Estimation:
Using predetermined Camera Calibration values, the homography is calculated and every point in image is mapped accurately to it's correspoinding 3D world coordinates. 


## Sample inference:
----

<img src="/readme_photos/sample_inference.gif" width="600"> 

## Notes
----
* Weights files and training datasets are not present in the repo
* Calibration values must be properly tuned depending on the individual scene (_Hint: Maybe will be automated in the future_)

## Contributing
----
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.