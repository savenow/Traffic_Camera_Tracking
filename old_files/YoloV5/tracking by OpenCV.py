import cv2
import numpy as np
import torch
import time
import random
import os
import torch.backends.cudnn as cudnn
import math


# Load Model and weights.
model_weights = r'C:\Users\ronak\Desktop\velocity_estimation\tl_yolo5l6_78k_bs_3.pt'
cap = cv2.VideoCapture(r'C:\Users\ronak\Desktop\velocity_estimation\test_1.mp4')
model = torch.hub.load(r'C:\Users\ronak\Desktop\velocity_estimation\Traffic_Camera_Tracking\YoloV5\yolo_v5_main_files',
                           'custom',
                           path=model_weights,
                           source='local')
device = torch.device('cuda:0')
cudnn.benchmark = True
# print(torch.cuda.is_available())


# Start of main Code...
classID_dict = {0: ("Escooter", (0, 90, 255)), 1: ("Pedestrians", (255, 90, 0)), 2: ("Cyclists", (90, 255, 0))}
count = 0
center_pts_prev_frame = []
tracking_objects = {}
track_id = 0

while True:
    _, frame = cap.read()
    count += 1
    if _:
        
        center_pts_cur_frame = []
        box_coord = {}
        results = model(frame, size=1920)
        result = results.xyxy[0]
        

        if len(result) > 0: 
            dets = []
            for items in result:
                # print(items[0:4])
                dets.append(items[0:7].tolist())
            #   dets = np.array(dets)
            for bbox in dets:
                (x1, y1, x2, y2, score, class_id) = bbox
                color = classID_dict[class_id][1]
                cx = int((int(x1) + int(x2))/2)
                cy = int((int(y1) + int(y2))/2)
                center_pts_cur_frame.append((cx, cy))
                box_coord[(cx, cy)] = [int(x1), int(y1), int(x2), int(y2)]
                label_2 = f'{classID_dict[class_id][0]} {round(score*100, 1)}%'
                (w2, h2), _ = cv2.getTextSize(label_2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - 20), (int(x1) + w2, int(y1)), color, -1)
                text_color = (0, 0, 0)
                cv2.putText(frame, label_2, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
            del color

            if count <= 2:
                for pt in center_pts_cur_frame:
                    for pt2 in center_pts_prev_frame:
                        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                        if distance < 20:
                            tracking_objects[track_id] = pt
                            track_id += 1
            else:
                tracking_objects_copy = tracking_objects.copy()
                center_pts_cur_frame_copy = center_pts_cur_frame.copy()
                for object_id, pt2 in tracking_objects_copy.items():
                    object_exists = False
                    for pt in center_pts_cur_frame_copy:
                        distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                        
                        # update object position
                        if distance < 20:
                            tracking_objects[object_id] = pt
                            object_exists = True
                            if pt in center_pts_cur_frame:
                                center_pts_cur_frame.remove(pt)
                                continue

                    # remove the id
                    if not object_exists:
                        tracking_objects.pop(object_id)

                for pt in center_pts_cur_frame:
                    tracking_objects[track_id] = pt
                    track_id += 1

            temp_pt = []
            for object_id, pt in tracking_objects.items():
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
                temp_pt.append([object_id, pt])

            

            for t in temp_pt:
                if t[1] in box_coord.keys():
                    x1 = box_coord[t[1]][0]
                    y1 = box_coord[t[1]][1]
                    label_1 = f'Track_ID: {str(t[0]+1)}'
                    (w1, h1), _ = cv2.getTextSize(label_1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    text_color = (0, 0, 0)
                    cv2.rectangle(frame, (x1, y1 - 40), (x1 + w1, y1-21), (0,255,255), -1)
                    cv2.putText(frame, label_1, (x1, y1 -24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

            

            # print(f'tracking_obkects : {tracking_objects}')
            # print(f'cur_frm : {center_pts_cur_frame}')
            # print(f'prev_frm : {center_pts_prev_frame}')

            
            cv2.imshow('frame', frame)
            # cv2.resizeWindow('frame', 1200, 1600)

            center_pts_prev_frame = center_pts_cur_frame.copy()

            key = cv2.waitKey(1)
            if key == ord('q'): 
                break
            if key == ord('p'):
                cv2.waitKey(-1)
    else:
        break
cap.release()
cv2.destroyAllWindows()