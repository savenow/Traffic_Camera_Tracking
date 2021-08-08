from sort import *
import random
import cv2

color_boxes = [(255, 99, 99), (255, 120, 120), (255, 150, 150), (255, 80, 80)]

def show_tracker_bbox_score(input, frame):
    for (detection, score) in input:
        # For bounding box
        tracker_id = detection[4]
        x1, x2, y1, y2 = detections[0:4]

        color = random.choice(color_boxes)
        img = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        label = f'Track_ID: {tracker_id}, {score}'
        (w, h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prints the text.    
        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        text_color = (0, 0, 0)
        img = cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # For printing text
        img = cv2.putText(img, 'test', (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        
        return img



Object_tracker = Sort()
# get detections
boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
scores = predictions.scores if predictions.has("scores") else None
if boxes is None:
    track_bbs_ids = mot_tracker.update()
else:
    detections = process_bboxes_scores(boxes, scores)
# detections is a np.array of the form [[x1, x2, y1, y2, score], [x1, x2, y1, y2, score], ....]

# update SORT
track_bbs_ids = mot_tracker.update(detections)
show_tracker_bbox_score(zip(detections, scores), frame)





# Predictions is the outputs of detectron2 and this manipulates data from it
# 

# classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
# masks = predictions.pred_masks.tensor.numpy()