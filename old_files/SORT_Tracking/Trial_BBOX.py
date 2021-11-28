from sort import *
import random
import cv2
import numpy as np

color_boxes = [(255, 99, 99), (255, 120, 120), (255, 150, 150), (255, 80, 80)]

def show_tracker_bbox_score(input, frame):
    
    img = frame
    for (detection, score) in input:
        # For bounding box
        print(f"Detection: {detection}, Score: {score}")
        tracker_id = detection[4]
        x1, y1, x2, y2 = detection[0:4]

        print(f"Tracker ID: {tracker_id}")
        print(f"X1: {x1}, Y1: {y1}, X2: {x2}, Y2: {y2}")

        color = random.choice(color_boxes)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        label = f'Track_ID: {tracker_id}, {score[0] * 100}%'
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prints the text.    
        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        text_color = (0, 0, 0)
        img = cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
    return img


img = np.zeros((512, 512, 3), np.uint8)
detections = np.array([[100, 100, 200, 200, 1], [100, 200, 200, 300, 2], [100, 300, 400, 400, 3], [50, 50, 100, 300, 4]])
scores = np.array([[0.99], [0.87], [0.8], [0.45]])
cv2.imshow('Image', show_tracker_bbox_score(zip(detections, scores), img))
cv2.waitKey(0)
