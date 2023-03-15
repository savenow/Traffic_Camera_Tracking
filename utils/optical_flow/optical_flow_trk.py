import cv2
import numpy as np
import torch
import torchvision as tv
import sys
sys.path.append('/home/mobilitylabextreme002/Desktop/Traffic_Camera_Tracking')
from deep_sort.deep_sort import DeepSort
from sort_yoloV5 import Sort
from visualizer import Visualizer, Minimap

sys.path.append('/home/mobilitylabextreme002/Desktop/Traffic_Camera_Tracking/yolo_v5_main_files')
from models.common import DetectMultiBackend, AutoShape
from utils.datasets import LoadImages
from utils.torch_utils import time_sync
from utils.general import LOGGER, non_max_suppression, \
    scale_coords, check_img_size, print_args, xyxy2xywh

IMG_HEIGHT = 1080
IMG_WIDTH = 1920
model = torch.hub.load(
    "/home/mobilitylabextreme002/Desktop/Traffic_Camera_Tracking/yolo_v5_main_files", "custom", 
    "/home/mobilitylabextreme002/Desktop/weights/All_5_combined/weights/best.pt", source="local"
)
model.amp = True

def addBBOXPadPixels(x, y, w, h):
    # Adds extra padding to bbox coordinate to ensure a loose crop
    # Also converts BBOX format from xywh to xyxy
    BBOX_PAD_PIXELS = 5
    if x - BBOX_PAD_PIXELS > 0:
        x1 = x - BBOX_PAD_PIXELS
    else:
        x1 = 0
    
    if y - BBOX_PAD_PIXELS > 0:
        y1 = y - BBOX_PAD_PIXELS
    else:
        y1 = 0
    
    if IMG_WIDTH - (x + w + BBOX_PAD_PIXELS) > 0:
        x2 = x + w + BBOX_PAD_PIXELS
    else:
        x2 = IMG_WIDTH
    
    if IMG_HEIGHT - (y + h + BBOX_PAD_PIXELS) > 0:
        y2 = y + h + BBOX_PAD_PIXELS
    else:
        y2 = IMG_HEIGHT

    return (x1, y1, x2, y2)

def getFrameDiffBBOX(framenumber, video_image, previous_frame):
    THRESH_VALUE = 10
    MINIMUM_BBOX_AREA = 300
    
    # 2. Prepare image; grayscale and blur
    prepared_frame = cv2.cvtColor(video_image, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=(5, 5), sigmaX=0)

    if framenumber == 1:
        previous_frame = prepared_frame
        return previous_frame, None
    else:
        # calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=previous_frame, src2=prepared_frame)
        previous_frame = prepared_frame

        # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        thresh_frame = cv2.threshold(src=diff_frame, thresh=THRESH_VALUE, maxval=255, type=cv2.THRESH_BINARY)[1]
        # 5. Only take different areas that are from visualizer import Visualizer, Minimap
        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        bbox_list = []
        for index, contour in enumerate(contours):
            if cv2.contourArea(contour) < MINIMUM_BBOX_AREA:
                # too small: skip!
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            bbox_list.append((x, y, x+w, y+h))

        return previous_frame, bbox_list

def FilterBBOXes(yolo_bboxes, frameDiff_bboxes):
    FILTER_OVERLAP_THRESH = 0.4
    yolo_xyxy = yolo_bboxes[..., :4]
    frameDiff_bboxes = torch.tensor(frameDiff_bboxes)
    # Calculate IoU between Bboxes
    iou = tv.ops.box_iou(yolo_xyxy, frameDiff_bboxes)

    # Select only frameDiff boxes which has IoU ovelap of less than FILTER_OVERLAP_THRESH with yolo bboxes
    max_tracker_overlap = torch.max(iou, dim=0)[0]
    gather_indices = torch.where(max_tracker_overlap < FILTER_OVERLAP_THRESH)[0]
    selected_frameDiffboxes = torch.index_select(frameDiff_bboxes, 0, gather_indices)

    # Add dummy columns for conf, classID with -1 for selected frameDiff boxes
    selected_frameDiffboxes = torch.cat((selected_frameDiffboxes, torch.ones(selected_frameDiffboxes.shape[0], 2)*-1.0), 1)
    final_bboxes = torch.cat((yolo_bboxes, selected_frameDiffboxes), 0)
    
    return final_bboxes

def UpdateTracker_deepSort(DeepSortObj, trackerlist, pred, im0):
    if len(pred) > 0:
        pred = pred.cpu()
        bbox_xywh = xyxy2xywh(pred[:, :4])
        confs = pred[:, 4:5]
        labels = pred[:, 5:6]
        output = DeepSortObj.update(
            bbox_xywh, confs, labels, im0
        )
        trackerlist = []
        for det in output:
            trackerlist.append([det[0], det[1], det[2], det[3], det[6], det[5], 0.0, 0.0, 0.0, det[4]])

    return trackerlist

def UpdateTracker(SortObj, trackerlist, pred):
    if len(pred) > 0:
        dets = []
        for items in pred:
            dets.append(items[:].tolist())
    
        dets = np.array(dets)
        # print(dets)
        # exit()
        trackerlist = SortObj.update(dets)
    else:
        trackerlist = SortObj.update()
    return trackerlist

def motion_detector(cap): 
    frame_count = 0
    previous_frame = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Specify the video codec
    out = cv2.VideoWriter('outputs/raghav_test_amp_opticalFlow.mp4', fourcc, 30.0, (1280, 720)) # Specify the output filename, codec, frame rate, and frame size
    #out_thresh = cv2.VideoWriter('outputs/32_thresh_opticalFlow.mp4', fourcc, 30.0, (1280, 720)) # Specify the output filename, codec, frame rate, and frame size
    Objtracker_DeepSort = DeepSort(
        "deep_sort/deep/checkpoint/ckpt.t7", 
        max_dist=0.2, max_iou_distance=0.7, max_age=70, 
        n_init=3, nn_budget=100, use_cuda=True
    )
    # Initialize Tracker
    Objtracker = Sort(
        max_age=70, 
        min_hits=5, 
        iou_threshold=0.25
    )
    Objtracker.reset_count()

    classID_dict = {
        -1: ("FrameDiff", (0, 0, 0)),
        0: ("Escooter", (0, 90, 255)), 
        1: ("Pedestrians", (255, 90, 0)), 
        2: ("Cyclists", (90, 255, 0)),
        3: ("Motorcycle", (204, 0, 102)),
        4: ("Car", (0, 0, 255)),
        5: ("Truck", (0, 102, 204)),
        6: ("Bus", (0, 255, 255))
    }
    # im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
    frameDiff_frame_prev = None
    tracker_list = []
    Visualize = Visualizer(True, True, 30, 250, 0)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            print(f"Processing frame: {frame_count}")

            yolo_results = model(frame[..., ::-1], size=1920).xyxy[0].cpu()
            # for x1, y1, x2, y2, conf, class_id in yolo_results.numpy():
            #     cv2.rectangle(img=frame, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=classID_dict[int(class_id)][1], thickness=2)

            frameDiff_frame_prev, frame_diff_bbox_list = getFrameDiffBBOX(frame_count, frame, frameDiff_frame_prev)
            if frame_diff_bbox_list: # Only non empty frame different bboxes
                # for x1, y1, x2, y2 in frame_diff_bbox_list:
                #     cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 0), thickness=3)

                final_results = FilterBBOXes(yolo_results, frame_diff_bbox_list)
                # for x1, y1, x2, y2, conf, class_id in final_results.numpy():
                #     cv2.rectangle(img=frame, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=classID_dict[int(class_id)][1], thickness=2)
                
                # tracker_list = UpdateTracker_deepSort(Objtracker_DeepSort, tracker_list, final_results, frame)
                tracker_list = UpdateTracker(Objtracker, tracker_list, final_results)
                
                if len(tracker_list) > 0:
                    frame = Visualize.drawTracker(tracker_list, frame, frame_count)
                elif len(final_results) > 0:
                    frame = Visualize.drawBBOX(final_results, frame, frame_count)
                else:
                    frame = Visualize.drawEmpty(frame, frame_count)
            # results = 
            # results.save()
            # print(results.xyxy)
            # exit()
            #exit()
            #cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            # cv2.resize(thresh_frame, (1280, 720))
            frame = cv2.resize(frame, (1280, 720))
            out.write(frame)
            # out_thresh.write(cv2.resize(cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR), (1280, 720)))
            # cv2.imshow("thresh", thresh_frame)
            # cv2.imshow("countour", img_rgb)
            
            # k = cv2.waitKey(10) & 0xFF
            # if k == 27:
            #     break

            if frame_count == 6000:
                break

        else:
            break
    
    out.release()

# vid 32.m4: /home/mobilitylabextreme002/Videos/small_clipped/32.mp4
# vid raghav: 
cap = cv2.VideoCapture("/home/mobilitylabextreme002/Videos/small_clipped/raghav_calibration/raghav_clipped.mp4")
motion_detector(cap)

